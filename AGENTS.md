# Rules for agents working on branch `guanyu` (vlm-em project)

This branch is Guanyu's vlm-em work: the yes-habit experiment in `experiments/yes_habit/`.
Every agent (Claude, Codex, ...) must follow these rules, on the RCP cluster and in local clones alike.

## On RCP: the team READMEs are binding
This clone lives on a **shared** RCP account. Read these two files, written by Zhihao, at the start of every session
and follow them strictly. They override this file:
- `/mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/README.md`: shared data, models and HF cache (read-only)
- `/mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/README_TEAM.md`: rules of the shared account

Also read Guanyu's `/mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/guanyu/CLAUDE.md` (project rules).
The experiment is closed (2026-11-25): results in `experiments/yes_habit/REPORT.md`, frozen criteria in
`experiments/yes_habit/proposals/`; the full plan and decision log are archived in `guanyu/archive/yes_habit_2026-11/PLAN.md`.

Key points, from those files:
- Write only in `guanyu/`.
- Shared `data/`, `models/` and `hf_cache/` are read-only.
- Teammates' folders and branches are read-only.
- Run:AI jobs are prefixed `guanyu-vlmem-`, and are submitted or deleted only with Guanyu's OK.
- Use only the `guanyu-secrets` secret, never `mem-secrets`.
- No computation on the login node.
- Never change shared account settings.

## Everywhere (RCP and local clones, e.g. Guanyu's Mac)
- **Git:** commit only on branch `guanyu`, with Guanyu's identity. Push only with Guanyu's explicit OK, after
  checking author and committer, the files and secrets, and a clean tree. Never force-push. Never touch other branches.
- **Science:** follow the frozen criteria in `experiments/yes_habit/proposals/`. Never change a pre-registered criterion after seeing its results. Record
  every deviation, such as the pilot's image-size cap, with its date.
- **Local agents (e.g. on the Mac):** run and report unless Guanyu asks for changes. Fixes are made in the RCP clone.
  Never print tokens or secrets.
