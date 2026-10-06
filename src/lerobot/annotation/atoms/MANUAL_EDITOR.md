# Manual atomic-subtask editor

From `/home/user/Documents/Research/RL/LeRobot`:

```bash
uv run python -m lerobot.annotation.atoms.manual_editor
```

Open **http://127.0.0.1:8766**. Use `--port 8767` if that port is occupied.
The server binds to this computer only. No packages, build step, model, or network
service are needed. Stop it with Ctrl+C in its terminal.

## Annotate an episode

1. Choose a source/task and an episode. The status filter can show only unfinished
   episodes. Existing agent reviews and manual overrides are loaded automatically;
   an unfinished episode starts with the current proposed cuts.
2. Select a parent subtask. Both camera views follow the same episode playhead.
   Select another available camera using the dropdown above either view.
3. Click an atom to seek to its start. Scrub or step to a physical action boundary,
   then press **S** to split. Edit the action, object and destination on the right.
   **Merge with next** keeps the selected label; **Undo** reverses an edit.
4. Adjust start/end frames directly, or use **[** / **]** to set them at the playhead.
   The adjacent atom moves with the boundary, preserving continuous coverage.
   Parent boundaries stay fixed. End frames are exclusive.
5. **Save draft** at any point. **Mark episode reviewed** validates all parents and
   saves a completed manual override. Errors appear above the timeline. Review
   confidence, notes, inherited parent events and optional quality overrides are
   available in the editor. For optional new events, click **Apply events JSON**
   after editing their JSON.

**Space** plays/pauses; **←/→** step a frame; **Shift+←/→** step a second;
**Ctrl+S** saves a draft; **Ctrl+Z** undoes an edit outside a text field.
**Play atom** stops at its boundary; check **Loop atom** to repeat it.
Playback rates are 0.25×, 0.5×, 1× and 2×.

Edits are also retained in this browser's local storage for recovery. Switching
episodes saves a dirty draft first. Disk saves are explicit or happen on episode
navigation; closing the tab with unsaved changes prompts the browser. Use Save
draft before stopping the server. An episode changed by another session is
rejected on save instead of silently overwritten.

## Files and pipeline

All saves go beneath `outputs/_annotation/subtask_atoms_review/`:

- `manual_drafts/<episode_id>.json`: resumable work, ignored by the collector.
- `manual_overrides/<episode_id>.json`: completed human reviews, taking precedence
  over existing agent reviews in `collect_reviews.py`.
- `manual_history/<episode_id>/`: timestamped backups before overwriting a save.

The editor checks the existing grammar, exact coverage of each parent, minimum
atom duration, quality rules and mistake events. A passing check is structural;
it does not establish visual correctness. Drafts may fail these checks.

The corpus videos and parent annotations are read-only. Marking reviewed does
not assemble atoms, recompute speed labels, or launch training. The next pipeline
step remains collection with `collect_reviews.py`, followed by `assemble.py`.
See the [review guide](REVIEW_GUIDE.md) for the retained review decisions.
FMB and ReBot are outside this editor's diverse-corpus scope.

For isolated experiments, use `--review-dir /tmp/my-atomic-reviews`; this also reads
existing reviews from that alternate directory, so it initially shows proposals
unless you copy reviews into it.

## Verification

```bash
uv run python -m lerobot.annotation.atoms.test_manual_editor
```

Tests use temporary review directories and cover draft/completed saves, validation,
backups, concurrent-save conflicts, restricted video access and byte-range seeking.
The browser test uses the already available Playwright Chromium and checks actual
video loading, frame stepping, playback, splitting, undo, boundary edits and
save/reload. It writes a screenshot to `/tmp/atomic-editor-smoke.png`.
