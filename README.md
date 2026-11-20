# Snap & Claim

A lost-and-found prototype for events. Finders submit a photo; owners describe what they lost, then answer a question about a distinguishing detail before the app shows a photo. Staff should verify ownership in person before releasing any item.

## User and task

Event attendees need to report or recover lost items. A finder takes a photo and notes where they found the item. Someone who lost an item searches in their own words and answers an ownership question. Front desk volunteers review claims, manage listings, and record returns.

## Benefit

Finders don’t have to write a listing from scratch, attendees don’t have to guess the exact words a finder used, and volunteers can look up the item and its claim history in one place.

## Services

| Akamai service | What it does and why it fits |
| --- | --- |
| Cloud Inference | Runs vision, embedding, and chat models without requiring a separate GPU host. |
| Managed PostgreSQL + pgvector | Stores item status, claim attempts, and 4,096-dimensional search vectors together. Staff updates and claim history persist across app restarts. |
| Object Storage | Holds photos in a private bucket, separate from searchable text. The app issues short-lived signed links to staff or after an AI-accepted claim. |
| Linode + Cloud Firewall | Hosts the FastAPI app and corkboard UI on a VM with restricted ingress for the workshop deployment. |

## Inference

| Deployed model | Task and reason for using it | How we checked it |
| --- | --- | --- |
| `gemma-4-26b-a4b-it` | Turns a finder’s photo into a public listing and a private clue and ownership question in a single vision call. | Reviewed a generated listing and question from a staged photo, checking that the question was shown without the clue. |
| `qwen3-embedding-8b` | Matches a free-text description to similar listings even when the wording differs. | Exercised the search UI against deployed inventory. Automated tests use fixed vectors to verify that exactly 50% is included and anything below it is excluded before rounding. |
| `qwen3-8-27b` | Compares an attendee’s free-text answer with the stored clues without requiring an exact phrase. | Checked a rejected answer in the app. Automated tests stub the verdict to exercise accepted and rejected attempts and verify that public feedback does not reveal private clues. |

These are workflow checks, not a model accuracy benchmark. An AI verdict is a signal for staff, not proof of ownership.

## Technical choice

For event-sized inventory, use an exact cosine-distance scan in PostgreSQL rather than operating a separate vector service. Rank only open items and apply the 50% similarity cutoff before rounding for display. Treat the score as a search signal, not the probability that the item belongs to the claimant. Staff search is a separate text lookup over inventory; it does not call a model.

## Our idea

A searchable lost-and-found board doesn’t need to show every photo or identifying detail. Snapclaim separates **finding a likely match** from **proving ownership**: attendees see descriptions and locked photos, while staff can review private clues and submitted answers before handing over an item.

## Demo

**Input:** A staged photo of a USB-C charger with a strip of green tape on its underside, found by a workshop table.

1. Choose **I found something**, add the photo and location, and pin the item. Review the generated description and ownership question. Use a staged photo where the tape is visible.
2. Choose **I lost something** and search for `small charger near the workshop tables`. The listing should appear as a possible match with its photo locked. Try an unrelated description to show how the board handles no close match.
3. Submit an incorrect answer, then a truthful distinguishing detail such as `green tape on the underside`. A rejected attempt keeps the photo locked; an accepted attempt reveals the photo and directs the claimant to the desk.
4. In **Front desk** (`/?staff`), find the listing and open **Show verification details**. Compare the clue with the submitted answers, verify ownership in person, then choose **Mark returned**. The returned item no longer appears in attendee search.

## Known limits

- Use staged photos for the prototype. Vision can miss or misread markings, and the answer judge can accept or reject incorrectly. The app can reveal a photo on an AI-accepted claim before staff review; human approval before disclosure is not implemented.
- The Front desk uses a shared PIN; there are no individual staff accounts or authenticated finder accounts. Broader rollout needs stronger abuse controls, per-staff access, and a retention policy.
- Exact vector search fits the pilot; retrieval quality, performance, and the 50% cutoff have not been validated at larger scale.

## Run locally

From the root of this repository, copy `.env.example` to `.env`, add your Akamai Inference key, and set a long random `DESK_PIN` if you want the staff view.

```sh
cp .env.example .env
chmod 600 .env
# Add the inference key in your local editor; never commit .env.
uv run --locked app.py
```

Open `http://localhost:8000`. The staff view is at `http://localhost:8000/?staff` and requires the configured PIN. Without `PG*` and `OBJ_*` settings the app uses in-memory development storage, which is lost on restart.

Staff can search titles, descriptions, found locations, and item IDs. Results update as they type and can be narrowed with the status filter. Switching away from the desk or reloading the page requires the PIN again. Staff can also edit public titles, descriptions and locations, mark physical returns, archive records, and reopen items. Only open items appear in attendee search. Archiving preserves the photo and claim history; reopening does not clear earlier claim attempts. Updates reject stale listings instead of overwriting another staff member's change.

On an item card, open **Show verification details** to compare the ownership question, private clues, and submitted answers. The app shows the AI outcome for each attempt, not a separate official answer. This data is loaded only on demand after PIN authentication and is not part of the public search or staff inventory list. Don’t share clues or past answers with claimants; verify ownership in person.

To use persistent storage, configure `PG*` for a PostgreSQL role and schema with `pgvector` available, verify TLS with your database CA, and configure `OBJ_*` with a private bucket and a bucket-scoped key. The example `snapclaim.service` expects the checkout at `/root/snapclaim` and binds to port 80; use it only on a dedicated host or after checking for a port conflict. Keep credentials out of Git and use a firewall to restrict ingress.
