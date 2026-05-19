# Engineering Notes

This folder is the repo-wide field manual for this LeRobot fork. It records implementation rationale, operational runbooks, debugging notes, and open questions that are too specific or provisional for polished user documentation.

The goal is to make the repo easier to reason about later: why choices were made, what assumptions are currently true, where important code paths connect, and what to check when something fails.

## Map

| Area | Start Here | Purpose |
| --- | --- | --- |
| Decisions | `decisions/` | Why a design or implementation choice was made. |
| Runbooks | `runbooks/` | What to do when running, debugging, or recovering from a known issue. |
| Architecture | `architecture/` | How major repo components connect. |
| Open questions | `open_questions.md` | Things we have not resolved yet. |

## Current High-Value Notes

| Topic | File | Why it matters |
| --- | --- | --- |
| Linux overcommit and probe MP4 failures | `runbooks/system_overcommit.md` | Explains `[Errno 12] Cannot allocate memory` from imageio/ffmpeg during attention probes. |

## Conventions

- Prefer short, focused notes over one giant document.
- Put root-cause/debugging procedures in `runbooks/`.
- Put rationale and tradeoffs in `decisions/`.
- Link to concrete files when a note depends on implementation details.
- Keep uncertainty explicit rather than pretending a hypothesis is settled.
