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
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import psycopg
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel

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
        results = sorted(({**{k: v for k, v in item.items() if k not in ("embedding", "private_details")},
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
    item_id: str
    answer: str


@app.post("/claim")
def claim(body: Claim):
    if not body.answer.strip():
        raise HTTPException(400, "Answer the question first.")
    if DATABASE:
        with db() as conn:
            row = conn.execute("select private_details, proof_question, photo_key, found_where, status, "
                               "(select count(*) from claims where item_id = items.id and not passed) "
                               "from items where id = %s", (body.item_id,)).fetchone()
    else:
        item = MEMORY["items"].get(body.item_id)
        fails = sum(1 for c in MEMORY["claims"] if c["item_id"] == body.item_id and not c["passed"])
        row = item and (item["private_details"], item["proof_question"], item["photo_key"],
                        item["found_where"], item["status"], fails)
    if not row:
        raise HTTPException(404, "Item not found.")
    private, question, key, where, status, fails = row
    if status != "open":
        raise HTTPException(409, "This item was already claimed.")
    if fails >= 3:
        raise HTTPException(429, "Too many wrong answers. Please see the front desk with ID.")
    verdict = judge_claim(private, question, body.answer.strip())
    record = {"item_id": body.item_id, "answer": body.answer.strip(), "passed": verdict["passed"],
              "confidence": verdict.get("confidence"), "reason": verdict.get("reason")}
    if DATABASE:
        with db() as conn:
            conn.execute("insert into claims (item_id, answer, passed, confidence, reason) values (%s,%s,%s,%s,%s)",
                         (body.item_id, record["answer"], record["passed"], record["confidence"], record["reason"]))
            if verdict["passed"]:
                conn.execute("update items set status = 'claimed' where id = %s", (body.item_id,))
    else:
        MEMORY["claims"].append(record)
        if verdict["passed"]:
            MEMORY["items"][body.item_id]["status"] = "claimed"
    if not verdict["passed"]:
        return {"passed": False, "reason": verdict.get("reason"), "tries_left": max(0, 2 - fails)}
    return {"passed": True, "reason": verdict.get("reason"), "photo_url": photo_link(key),
            "pickup": f"Show this screen at the front desk. It was found at: {where or 'not noted'}."}


DESK_PIN = os.environ.get("DESK_PIN", "")


@app.get("/desk")
def desk(x_desk_pin: str = Header("")):
    """Staff-only finder-desk view: every item with its status and a short-lived photo link.
    Photos are the secret owners prove against, so this view needs the DESK_PIN from .env."""
    if not DESK_PIN:
        raise HTTPException(403, "Front desk is off. Set DESK_PIN in .env to turn it on.")
    if x_desk_pin != DESK_PIN:
        raise HTTPException(401, "Wrong staff PIN.")
    if DATABASE:
        with db() as conn:
            rows = conn.execute("select id, title, found_where, status, photo_key, created_at, "
                                "(select count(*) from claims where item_id = items.id) from items "
                                "order by created_at desc limit 50").fetchall()
        items = [{"id": str(r[0]), "title": r[1], "found_where": r[2], "status": r[3], "photo_url": photo_link(r[4]),
                  "created_at": r[5].isoformat(), "claims": r[6]} for r in rows]
    else:
        items = [{"id": i["id"], "title": i["title"], "found_where": i["found_where"], "status": i["status"],
                  "photo_url": photo_link(i["photo_key"]), "created_at": i["created_at"],
                  "claims": sum(1 for c in MEMORY["claims"] if c["item_id"] == i["id"])}
                 for i in sorted(MEMORY["items"].values(), key=lambda i: i["created_at"], reverse=True)]
    return items


@app.get("/dev-photo/{key:path}")
def dev_photo(key: str):
    if s3 or key not in MEMORY["photos"]:
        raise HTTPException(404)
    return Response(MEMORY["photos"][key], media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=int(os.environ.get("PORT", 8000)))
