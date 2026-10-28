# Deployment

The same Docker image runs everywhere; it listens on `$PORT`. Offline mode works with no keys, so a
public demo costs nothing unless you add an LLM key (then set a spend limit with your provider and keep
`RATE_LIMIT_PER_MIN` on).

## Security profiles
| Profile | Settings | Use for |
|---|---|---|
| Open demo | defaults (`API_KEY` unset), `MANAGER_API_KEY` + `USER_TOKEN_SECRET` set, offline backend or a provider key with a spend cap | public link on a resume; reviewers can open `/docs` |
| Private / production | `APP_ENV=production`, `API_KEY`, `MANAGER_API_KEY`, `USER_TOKEN_SECRET`; the app refuses to boot if any is missing | anything with real users or real data |

Always set `TRUSTED_PROXY_HOPS` to the number of reverse proxies in front of the app (Render: 1), or
every user shares one rate-limit bucket. `/wait-time` is intentionally public (store-level numbers,
no user data) but is still rate limited. The limiter is per process; use a gateway or Redis for
more than one replica.

## Render (simplest, free tier)
1. Push to GitHub. In Render: **New > Blueprint** and select the repo (`render.yaml` is picked up).
2. Optional: add `ANTHROPIC_API_KEY` in the dashboard. `MANAGER_API_KEY` and `USER_TOKEN_SECRET` are generated for you.
3. Open `https://<service>.onrender.com/docs`.

## Google Cloud Run
```bash
gcloud run deploy qsr-assistant --source . --region asia-south1 --allow-unauthenticated \
  --set-env-vars RATE_LIMIT_PER_MIN=60 --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

## Hugging Face Spaces (Streamlit UI)
Create a Docker Space, push this repo, and add to the top of the Space's README:
```yaml
---
title: QSR GenAI Assistant
sdk: docker
app_port: 7860
---
```
Set the Space variable `PORT=7860`. For the UI instead of the API, override the command with
`streamlit run src/qsr_assistant/app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0`.

## With a vector database
`docker compose --profile qdrant up`, then in `.env`: `VECTOR_STORE=qdrant`, `QDRANT_URL=http://qdrant:6333`.

## Before calling it production
Derive `mode` from real authentication (not a client field), move sessions and the cache to Redis, put the
API behind a gateway with proper rate limiting, and rotate keys. See "Known limitations" in the README.

*Status: configs are provided and follow each platform's documented format, but this repository has not been
deployed from the author's sandbox. Deploy it once and put the live URL at the top of the README.*
