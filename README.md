# Snap & Claim

A lost-and-found prototype for events. Finders submit a photo; owners describe what they lost, then answer a question about a distinguishing detail before the app shows a photo. Staff should verify ownership in person before releasing any item.

## How it uses Akamai Cloud

- **Cloud Inference:** Gemma vision proposes a public description and a private proof question. Qwen embeddings rank search results. A Qwen chat model evaluates proposed answers; that verdict is a lead, not proof of identity.
- **Managed PostgreSQL + pgvector:** Stores inventory, claim attempts, and 4,096-dimensional embeddings. Search uses an exact cosine-distance scan.
- **Object Storage:** Stores photos in a private bucket. The app uses short-lived signed links for its protected desk view and after a passing claim.
- **Linode:** The included systemd unit is an example for a dedicated, firewall-restricted host.

## Run locally

From the root of this repository, copy `.env.example` to `.env`, add your Akamai Inference key, and set a long random `DESK_PIN` if you want the staff view.

```sh
cp .env.example .env
chmod 600 .env
# Add the inference key in your local editor; never commit .env.
uv run --locked app.py
```

Open `http://localhost:8000`. The staff view is at `http://localhost:8000/?staff` and requires the configured PIN. Without `PG*` and `OBJ_*` settings the app uses in-memory development storage, which is lost on restart.

Staff can search titles, descriptions, found locations, and item IDs, and combine their search with the status filter. Staff can also edit public titles, descriptions and locations, mark physical returns, archive records, and reopen items. Only open items appear in attendee search. Archiving preserves the photo and claim history; reopening does not clear earlier claim attempts. Updates reject stale listings instead of overwriting another staff member's change.

To use persistent storage, configure `PG*` for a PostgreSQL role and schema with `pgvector` available, verify TLS with your database CA, and configure `OBJ_*` with a private bucket and a bucket-scoped key. The example `snapclaim.service` expects the checkout at `/root/snapclaim` and binds to port 80; use it only on a dedicated host or after checking for a port conflict. Keep credentials out of Git and use a firewall to restrict ingress.

## Demo with staged objects

1. Submit a photo of a staged item, such as a charger with green tape. Check the public description and private proof prompt.
2. Search for the item in plain language; candidates should not show photos. Try an unrelated item to see the no-match state.
3. Try an incorrect ownership answer, then the known distinguishing detail. Staff must verify the match before releasing the physical item.

**Prototype limits:** Use staged photos only. A shared staff PIN and AI-assisted claims are not sufficient for an internet-facing service. Before broader use, add individual staff accounts, authenticated intake, human approval before photo disclosure, abuse controls, and a retention policy.
