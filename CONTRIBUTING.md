# Contributing to the AutoML backend

## The actual deploy flow

This repo's `main` branch is mirrored automatically to a live Hugging
Face Space via `.github/workflows/sync-to-hf.yml`. That means:

1. You fork this repo and open a PR against `main`.
2. A maintainer reviews and merges it.
3. The moment it merges, GitHub Actions pushes `main` to the HF Space,
   which rebuilds the Docker image and redeploys — for real, not a demo.

There's no separate "deploy step" a maintainer has to remember to run.
If your PR is merged, your change is live within a few minutes (Docker
builds with `torch` in them aren't instant).

## Before opening a PR

- Run `python -m py_compile` (or just try importing) on any file you
  touch — a broken import fails the deploy for everyone, not just you.
- If you change a phase's function signature in `src/phases/`, update
  `src/orchestrator.py`'s call to it in the same PR — they're not
  independently testable without a full pipeline run.
- If you touch `app.py`'s websocket event shapes (adding/renaming a
  field in any `progress(...)` call), the frontend repo's
  `lib/types.ts` and `lib/pipelineReducer.ts` need a matching update —
  flag this in your PR description so the frontend maintainer knows.
- Test locally first: `uvicorn app:app --reload --port 8000` and run an
  actual CSV through it. This project doesn't have an automated test
  suite yet (a genuine gap — a good first PR for anyone reading this).

## Repo secrets a maintainer needs configured (not something contributors set)

For the sync workflow to work, the repo's Settings → Secrets and
variables → Actions needs:
- `HF_TOKEN` — an HF access token with write access to the Space
- `HF_USERNAME` — the HF account/org the Space lives under
- `HF_SPACE_NAME` — the Space's name

## Local setup

See the main `README.md` for the full `venv` + `pip install` +
`uvicorn` setup.
