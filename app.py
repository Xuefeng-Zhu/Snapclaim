# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.162", "uvicorn>=0.54", "python-multipart>=0.0.33", "openai>=3.51,<4",
#                 "psycopg[binary]>=3.3", "boto3>=1.44", "python-dotenv>=1.2.4"]
# ///
"""Snap & Claim: a lost-and-found where finders snap a photo and owners prove it's theirs.

    uv run snapclaim/app.py      then open http://localhost:8000

Finder desk: photo -> VISION_MODEL writes a public summary plus private details -> photo to Object Storage,
             public text embedded with EMBEDDING_MODEL -> row in PostgreSQL (pgvector).
Owner:       describes the item -> vector search -> sees only public summaries (no photos)
             -> answers a proof question -> CHAT_MODEL judges against the private details
             -> on a pass, a 10-minute presigned photo link and pickup info.

Inference is required. Without PG*/OBJ_* in .env it runs in memory (dev mode), so you can build before the labs.
"""
import base64
import hmac
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

import boto3
import psycopg
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, Response as ApiResponse, UploadFile
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, StringConstraints

load_dotenv(override=True)
for name in ("AKAMAI_BASE_URL", "AKAMAI_API_KEY", "CHAT_MODEL", "VISION_MODEL", "EMBEDDING_MODEL"):
    if not os.environ.get(name):
        sys.exit(f"{name} is not set. Add it to .env and run uv run check.py.")

client = OpenAI(base_url=os.environ["AKAMAI_BASE_URL"], api_key=os.environ["AKAMAI_API_KEY"])
CHAT, VISION, EMBED = os.environ["CHAT_MODEL"], os.environ["VISION_MODEL"], os.environ["EMBEDDING_MODEL"]
DATABASE = bool(os.environ.get("PGHOST"))
BUCKET = os.environ.get("OBJ_BUCKET")
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("OBJ_ENDPOINT"),
    aws_access_key_id=os.environ.get("OBJ_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("OBJ_SECRET_KEY"),
    config=Config(signature_version="s3v4", request_checksum_calculation="when_required"),
) if BUCKET else None

MAX_PHOTO = 4_000_000          # bytes; the page shrinks photos to ~1024px JPEG before upload
MATCH_DISTANCE = 0.50          # Keep at least 50% cosine similarity before rounding for display.
LINK_SECONDS = 600             # presigned photo links last 10 minutes
app = FastAPI()

# ---------- storage: PostgreSQL + Object Storage, or memory in dev mode ----------
MEMORY = {"items": {}, "claims": [], "photos": {}}
MEMORY_LOCK = RLock()
_ready = False


def db():
    global _ready
    conn = psycopg.connect(connect_timeout=5)
    if not _ready:
        conn.execute("create extension if not exists vector")
        conn.execute("""create table if not exists items (
            id uuid primary key, title text not null, category text, public_text text not null,
            private_details text not null, proof_question text not null, found_where text, finder_note text,
            photo_key text not null, embedding vector(4096) not null, status text not null default 'open',
            created_at timestamptz not null default now())""")
        conn.execute("""create table if not exists claims (
            id bigserial primary key, item_id uuid references items(id), answer text not null,
            passed boolean not null, confidence real, reason text, created_at timestamptz not null default now())""")
        conn.commit()
        _ready = True
    return conn


def put_photo(key, data):
    if s3:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="image/jpeg")
    else:
        MEMORY["photos"][key] = data


def photo_link(key):
    if s3:  # bucket stays private; one item's photo, readable for 10 minutes
        return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=LINK_SECONDS)
    return f"/dev-photo/{key}"


def vec_text(v):
    return "[" + ",".join(map(str, v)) + "]"


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


# ---------- inference ----------
def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"model did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def describe_photo(jpeg):
    """Gemma looks at the photo. Public fields help owners find it; private fields are the proof."""
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    prompt = (
        "You catalog items for a lost-and-found. Look at the photo and reply with JSON only:\n"
        '{"title": "short generic name, e.g. Black phone charger",\n'
        ' "category": "one of: electronics, bag, clothing, bottle, keys, wallet, jewelry, document, other",\n'
        ' "public_text": "1-2 sentences an owner would recognize: type, main color, size, brand if obvious. '
        'Leave out unique marks.",\n'
        ' "private_details": "every unique detail only the real owner would know: stickers, scratches, text, '
        'contents, accessories, wear, exact colors",\n'
        ' "proof_question": "one question about a private detail, without revealing the answer"}\n'
        "If the photo shows no identifiable item, set title to \"unclear photo\"."
    )
    response = client.chat.completions.create(
        model=VISION,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        max_tokens=500, temperature=0.2,
    )
    return parse_json(response.choices[0].message.content)


def embed(text, query=False):
    """Qwen3-Embedding works best when the search side carries a one-line task instruction."""
    if query:
        text = f"Instruct: Given an owner's description of a lost item, retrieve the matching found item\nQuery: {text}"
    return client.embeddings.create(model=EMBED, input=[text]).data[0].embedding


def judge_claim(private_details, question, answer):
    """Qwen compares the owner's answer with what the photo showed. JSON so the app can act on it."""
    response = client.chat.completions.create(
        model=CHAT,
        messages=[
            {"role": "system", "content": (
                "You verify lost-and-found claims. The claimant answers a question, but the question is only a prompt: "
                "pass if the answer correctly names ANY specific detail from the private details (even one the question "
                "did not ask about) and states nothing that contradicts them. Fail vague answers (e.g. 'it is mine', "
                "'black') and answers with a wrong detail. Your reason must not mention, hint at, or correct any private "
                "detail; say only whether a specific matching detail was given. "
                'Reply with JSON only: {"passed": true|false, "confidence": 0-1, "reason": "one short sentence, '
                'no private details"} /no_think')},
            {"role": "user", "content": f"Private details: {private_details}\nQuestion asked: {question}\n"
                                        f"Claimant's answer: {answer}"},
        ],
        max_tokens=200, temperature=0,
    )
    verdict = parse_json(response.choices[0].message.content)
    verdict["passed"] = bool(verdict.get("passed"))
    return verdict


# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    status = {"inference": "configured", "database": "memory (dev)", "object_storage": "memory (dev)"}
    if DATABASE:
        try:
            with db() as conn:
                conn.execute("select 1")
            status["database"] = "connected"
        except Exception as error:
            status["database"] = f"error: {str(error).splitlines()[0][:150]}"
    if s3:
        try:
            s3.head_bucket(Bucket=BUCKET)
            status["object_storage"] = "connected"
        except Exception as error:
            status["object_storage"] = f"error: {str(error).splitlines()[0][:150]}"
    return status


@app.post("/items")
async def add_item(photo: UploadFile, found_where: str = Form(""), finder_note: str = Form("")):
    data = await photo.read(MAX_PHOTO + 1)
    if len(data) > MAX_PHOTO:
        raise HTTPException(413, "Photo is too large.")
    info = describe_photo(data)
    if info.get("title", "").lower() == "unclear photo":
        raise HTTPException(422, "The model couldn't make out an item. Retake the photo closer, in better light.")
    item_id = str(uuid.uuid4())
    key = f"items/{item_id}.jpg"
    put_photo(key, data)
    private = info["private_details"] + (f" Finder note: {finder_note}" if finder_note else "")
    search_text = f"{info['title']}. {info['category']}. {info['public_text']} Found at: {found_where}"
    vector = embed(search_text)
    row = {"id": item_id, "title": info["title"], "category": info.get("category", "other"),
           "public_text": info["public_text"], "private_details": private, "proof_question": info["proof_question"],
           "found_where": found_where, "finder_note": finder_note, "photo_key": key, "status": "open",
           "created_at": datetime.now(timezone.utc).isoformat()}
    if DATABASE:
        with db() as conn:
            conn.execute("""insert into items (id, title, category, public_text, private_details, proof_question,
                found_where, finder_note, photo_key, embedding) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                         (item_id, row["title"], row["category"], row["public_text"], private,
                          row["proof_question"], found_where, finder_note, key, vec_text(vector)))
    else:
        MEMORY["items"][item_id] = {**row, "embedding": vector}
    return {"id": item_id, "title": row["title"], "category": row["category"], "public_text": row["public_text"],
            "proof_question": row["proof_question"], "private_details": private}  # finder desk sees all


class Search(BaseModel):
    query: str


@app.post("/search")
def search(body: Search):
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Describe what you lost.")
    vector = embed(query, query=True)
    if DATABASE:
        with db() as conn:
            rows = conn.execute("""select id, title, category, public_text, found_where, proof_question, created_at,
                embedding <=> %s::vector as distance from items where status = 'open'
                order by distance limit 5""", (vec_text(vector),)).fetchall()
        results = [{"id": str(r[0]), "title": r[1], "category": r[2], "public_text": r[3], "found_where": r[4],
                    "proof_question": r[5], "created_at": r[6].isoformat(), "distance": float(r[7])} for r in rows]
    else:
        results = sorted(({**{k: v for k, v in item.items() if k not in ("embedding", "private_details", "_revision")},
                           "distance": cosine_distance(vector, item["embedding"])}
                          for item in MEMORY["items"].values() if item["status"] == "open"),
                         key=lambda r: r["distance"])[:5]
    confident = [r for r in results if r["distance"] <= MATCH_DISTANCE]
    for r in confident:
        r["match"] = round(max(0.0, 1 - r["distance"]) * 100)
        r.pop("photo_key", None); r.pop("finder_note", None)
    return {"results": confident, "near_misses": [],
            "message": None if confident else "Nothing turned in matches closely yet. Check back later."}


class Claim(BaseModel):
    item_id: uuid.UUID
    answer: str


def claim_snapshot(conn, item_id, *, lock=False):
    # Count attempts in a separate statement after taking the lock so a waiter's
    # count includes attempts committed by the previous holder.
    suffix = " for update" if lock else ""
    row = conn.execute("select private_details, proof_question, photo_key, found_where, "
                       "status, xmin::text from items where id = %s" + suffix, (item_id,)).fetchone()
    if row:
        count = conn.execute("select count(*) from claims where item_id = %s and not passed", (item_id,)).fetchone()[0]
        return (*row, count)
    return None


def memory_claim_snapshot(item_id):
    item = MEMORY["items"].get(item_id)
    if item:
        fails = sum(1 for c in MEMORY["claims"] if c["item_id"] == item_id and not c["passed"])
        return (item["private_details"], item["proof_question"], item["photo_key"], item["found_where"],
                item["status"], str(item.get("_revision", 0)), fails)
    return None


def check_claim(row, expected_revision=None):
    if not row:
        raise HTTPException(404, "Item not found.")
    if row[4] != "open" or (expected_revision is not None and row[5] != expected_revision):
        raise HTTPException(409, "This item changed or is no longer open. Search the board again.")
    if row[6] >= 3:
        raise HTTPException(429, "Too many wrong answers. Please see the front desk with ID.")


@app.post("/claim")
def claim(body: Claim):
    if not body.answer.strip():
        raise HTTPException(400, "Answer the question first.")
    item_id = str(body.item_id)
    if DATABASE:
        with db() as conn:
            row = claim_snapshot(conn, item_id)
    else:
        with MEMORY_LOCK:
            row = memory_claim_snapshot(item_id)
    check_claim(row)
    assert row is not None
    private, question, key, where, _, revision, _ = row
    verdict = judge_claim(private, question, body.answer.strip())
    record = {"item_id": item_id, "answer": body.answer.strip(), "passed": verdict["passed"],
              "confidence": verdict.get("confidence"), "reason": verdict.get("reason")}
    if DATABASE:
        with db() as conn:
            current = claim_snapshot(conn, item_id, lock=True)
            check_claim(current, revision)
            assert current is not None
            fails = current[6]
            conn.execute("insert into claims (item_id, answer, passed, confidence, reason) values (%s,%s,%s,%s,%s)",
                         (item_id, record["answer"], record["passed"], record["confidence"], record["reason"]))
            if verdict["passed"]:
                conn.execute("update items set status = 'claimed' where id = %s", (item_id,))
    else:
        with MEMORY_LOCK:
            current = memory_claim_snapshot(item_id)
            check_claim(current, revision)
            assert current is not None
            fails = current[6]
            MEMORY["claims"].append(record)
            if verdict["passed"]:
                MEMORY["items"][item_id].update(status="claimed", _revision=int(revision) + 1)
    if not verdict["passed"]:
        return {"passed": False, "reason": verdict.get("reason"), "tries_left": max(0, 2 - fails)}
    return {"passed": True, "reason": verdict.get("reason"), "photo_url": photo_link(key),
            "pickup": f"Show this screen at the front desk. It was found at: {where or 'not noted'}."}


DESK_PIN = os.environ.get("DESK_PIN", "")


def require_staff(x_desk_pin: str = Header("")):
    if not DESK_PIN:
        raise HTTPException(403, "Front desk is off. Set DESK_PIN in .env to turn it on.")
    if not hmac.compare_digest(x_desk_pin.encode(), DESK_PIN.encode()):
        raise HTTPException(401, "Wrong staff PIN.")


DeskStatus = Literal["open", "claimed", "returned", "archived"]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9]{1,20}$")]
TRANSITIONS = {"open": {"returned", "archived"}, "claimed": {"returned", "open", "archived"},
               "returned": {"open", "archived"}, "archived": {"open"}}


class DeskEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: Revision
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    public_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)]
    found_where: Annotated[str, StringConstraints(strip_whitespace=True, max_length=180)]


class DeskChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: Revision
    status: Literal["open", "returned", "archived"]


def check_staff_revision(current, requested):
    if current != requested:
        raise HTTPException(409, "Someone changed this item. Refresh the desk and retry.")


def memory_desk_item(item_id):
    item = MEMORY["items"].get(item_id)
    if not item:
        raise HTTPException(404, "Item not found.")
    return item


@app.get("/desk", dependencies=[Depends(require_staff)])
def desk(response: ApiResponse, status: DeskStatus | None = None, offset: int = Query(0, ge=0, le=10_000),
         q: str = Query("", max_length=160)):
    """Staff inventory, including archived records and short-lived photo links."""
    response.headers["Cache-Control"] = "no-store"
    q = q.strip()
    if DATABASE:
        with db() as conn:
            rows = conn.execute("select id, title, found_where, status, photo_key, created_at, public_text, xmin::text, "
                                "(select count(*) from claims where item_id = items.id) from items "
                                "where (%s::text is null or status = %s) "
                                "and (%s = '' or strpos(lower(concat_ws(' ', title, found_where, public_text, id::text)), lower(%s)) > 0) "
                                "order by created_at desc, id limit 50 offset %s", (status, status, q, q, offset)).fetchall()
        items = [{"id": str(r[0]), "title": r[1], "found_where": r[2], "status": r[3], "photo_url": photo_link(r[4]),
                  "created_at": r[5].isoformat(), "public_text": r[6], "revision": r[7], "claims": r[8]} for r in rows]
    else:
        def matches(item):
            if status is not None and item["status"] != status:
                return False
            text = " ".join(str(item[k]) for k in ("title", "found_where", "public_text", "id") if item[k] is not None)
            return not q or q.lower() in text.lower()

        with MEMORY_LOCK:
            rows = sorted((i for i in MEMORY["items"].values() if matches(i)),
                          key=lambda i: (i["created_at"], i["id"]), reverse=True)[offset:offset + 50]
            items = [{"id": i["id"], "title": i["title"], "found_where": i["found_where"], "status": i["status"],
                      "photo_url": photo_link(i["photo_key"]), "created_at": i["created_at"], "public_text": i["public_text"],
                      "revision": str(i.get("_revision", 0)),
                      "claims": sum(1 for c in MEMORY["claims"] if c["item_id"] == i["id"])} for i in rows]
    return items


@app.patch("/desk/items/{item_id}", dependencies=[Depends(require_staff)])
def edit_item(item_id: uuid.UUID, body: DeskEdit):
    key = str(item_id)
    if DATABASE:
        with db() as conn:
            candidate = conn.execute("select category, xmin::text from items where id=%s", (key,)).fetchone()
        if not candidate:
            raise HTTPException(404, "Item not found.")
        category, revision = candidate
    else:
        with MEMORY_LOCK:
            item = memory_desk_item(key)
            category, revision = item["category"], str(item.get("_revision", 0))
    check_staff_revision(revision, body.revision)
    vector = embed(f"{body.title}. {category}. {body.public_text} Found at: {body.found_where}")
    if DATABASE:
        with db() as conn:
            changed = conn.execute("update items set title=%s, public_text=%s, found_where=%s, embedding=%s::vector "
                                   "where id=%s and xmin::text=%s returning status, xmin::text",
                                   (body.title, body.public_text, body.found_where, vec_text(vector), key, revision)).fetchone()
            if not changed:
                raise HTTPException(409, "Someone changed this item. Refresh the desk and retry.")
    else:
        with MEMORY_LOCK:
            item = memory_desk_item(key)
            check_staff_revision(str(item.get("_revision", 0)), revision)
            item.update(title=body.title, public_text=body.public_text, found_where=body.found_where,
                        embedding=vector, _revision=int(revision) + 1)
            changed = item["status"], str(item["_revision"])
    return {"id": key, "status": changed[0], "revision": changed[1]}


def check_transition(status, revision, body):
    check_staff_revision(revision, body.revision)
    if body.status not in TRANSITIONS.get(status, set()):
        raise HTTPException(409, "This action is not available for the item's current status.")


@app.post("/desk/items/{item_id}/status", dependencies=[Depends(require_staff)])
def change_item_status(item_id: uuid.UUID, body: DeskChange):
    key = str(item_id)
    if DATABASE:
        with db() as conn:
            current = conn.execute("select status, xmin::text from items where id=%s for update", (key,)).fetchone()
            if not current:
                raise HTTPException(404, "Item not found.")
            current_status, current_revision = current
            check_transition(current_status, current_revision, body)
            updated = conn.execute("update items set status=%s where id=%s returning xmin::text", (body.status, key)).fetchone()
            assert updated is not None
            revision = updated[0]
    else:
        with MEMORY_LOCK:
            item = memory_desk_item(key)
            check_transition(item["status"], str(item.get("_revision", 0)), body)
            item.update(status=body.status, _revision=int(body.revision) + 1)
            revision = str(item["_revision"])
    return {"id": key, "status": body.status, "revision": revision}


@app.get("/dev-photo/{key:path}")
def dev_photo(key: str):
    if s3 or key not in MEMORY["photos"]:
        raise HTTPException(404)
    return Response(MEMORY["photos"][key], media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=int(os.environ.get("PORT", 8000)))
