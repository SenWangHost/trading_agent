---
name: deploy-to-droplet
description: Use when deploying the trading_agent to production, shipping merged changes live, updating or restarting the agent on the DigitalOcean droplet, or checking what code/version is currently running in prod.
---

# Deploy trading_agent to the DigitalOcean droplet

## Overview

Production runs as a single Docker Compose service on an Ubuntu droplet in
`schedule` mode (`restart: unless-stopped`). Deploying = get the latest `main`
onto the host, rebuild the image, and recreate the container. Secrets live in
`/root/trading_agent/.env` on the host (git-ignored, never redeployed).

- **Host:** `ssh root@159.223.169.140`
- **Repo path:** `/root/trading_agent`
- **Service:** `trading-agent` (container `trading_agent-trading-agent-1`)

## Before you deploy

**Deploying restarts a live trading container.** Confirm with the user first
unless they've already told you to ship. Check what's live first:

```bash
ssh root@159.223.169.140 'cd /root/trading_agent && git rev-parse --short HEAD && git branch --show-current && docker compose ps'
```

The host branch can lag `main` (it has been left on stale feature branches
before). Deploy always targets `main` — don't assume the host is on it.

## Deploy

Run as one SSH command so it's atomic and repeatable:

```bash
ssh root@159.223.169.140 'set -e && cd /root/trading_agent && \
  git fetch origin && \
  git checkout main && \
  git reset --hard origin/main && \
  docker build -t trading-agent:latest . && \
  docker compose up -d && \
  docker compose ps'
```

- `git reset --hard origin/main` discards any local host edits — intended; the
  host is a deploy target, not a workspace. `.env` is git-ignored so it survives.
- `docker compose up -d` recreates the container only because the image changed.
- Logs persist in the named `logs` volume across restarts.

## Verify

```bash
ssh root@159.223.169.140 'cd /root/trading_agent && \
  git rev-parse --short HEAD && \
  docker compose ps && \
  docker compose logs --tail 30 trading-agent'
```

- `docker compose ps` → status `Up`.
- Confirm the printed commit matches the `main` you expected to ship.
- Logs should show scheduler lines (`scheduler running — market hours...`) or a
  cycle in progress. Force a one-off cycle to smoke-test end to end:

```bash
ssh root@159.223.169.140 'cd /root/trading_agent && docker compose exec -T trading-agent uv run python main.py once'
```

## Rollback

Redeploy a known-good commit:

```bash
ssh root@159.223.169.140 'cd /root/trading_agent && \
  git checkout <good-sha> && docker build -t trading-agent:latest . && docker compose up -d'
```

## Gotchas

- **Droplet clock is UTC**; the scheduler pins `America/New_York` internally, so
  cycles still fire at correct ET times. Don't "fix" the host clock.
- **`.env` is not in git.** If it's ever missing, the container crash-loops on a
  missing key — copy it up with `scp .env root@159.223.169.140:/root/trading_agent/.env`.
- **Image tag is `:latest`** and built on-host, so `docker compose up -d` only
  recreates the container when the rebuild actually changed the image.
