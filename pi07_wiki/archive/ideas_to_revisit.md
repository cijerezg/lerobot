# Ideas to revisit

## From the pi0.7 paper (arXiv 2604.15483, read 2026-07-18)

- Mistake metadata belongs at subtask-segment granularity (their actual design; human-annotated boolean per action segment). Our subtask annotation chain already produces the segments to hang these on.
- Speed steering at inference: they prompt the per-task 15th percentile of training episode lengths — assumes constant work per episode; our variable-pairs task needs work-normalized speed (per-segment duration) instead.
- Classifier-free guidance on the metadata clause (beta 1.3-2.2) at inference to push actions toward the quality-5 / no-mistake mode. Cheap to try once metadata is enabled.
- Their headline ablation: without metadata in the prompt, adding lower-quality data degrades the model; with it, more data keeps helping. Enable metadata BEFORE mixing in rollout data.
- Their HL policy is not MEM's recurrent summary: obs + task + history of past subtask instructions -> next subtask, trained from verbal-coaching episodes. Both designs coexist in the pi lineage; doesn't settle our decode-order question.

## CANDIDATE KEY INNOVATION: ledger memory — "what was done / what needs to be done" (2026-07-18)

Memory-first decode where m_t is a two-part ledger: (done; remaining). At t=0 the first decode writes the initial scene reading + plan ("nothing done, need: pair white socks, pair black socks") — the cold start becomes a feature, not a gap. Each HL step moves items across the ledger; the subtask is nearly a pop off the head of "remaining."

Why it's a real delta: MEM's summary is retrospective-only, pi0.7's subtask history is retrospective-only; a recurrently GENERATED memory with a prospective half is new in this lineage (text plans exist in agent literature, but not as CE-trained recurrent VLA memory).

Practical wins: (1) supervision is free — "done" = past segments (current summary_annotate), "remaining" = hindsight compression of future canonical segments, same annotations, one prompt change; (2) crisp eval — does "remaining" shrink correctly?

Caveats: prospective half must update rather than persist when rollouts diverge from the demo plan; memory-first + ledger = more HL tokens, no early-exit — measure the latency, don't guess it.

## Summary-first vs subtask-first HL decode order (2026-07-18)

Current seam decodes `<subtask>. Memory: <summary>` — subtask conditions on stale m_t, summary decoded after. Alternative: `Memory: <m_{t+1}>. Next: <subtask>` — update memory from what just happened, then pick the subtask conditioned on the fresh summary (summary acts as chain-of-thought; subtask can't contradict memory).

RESOLVED 2026-07-18 (re-read the paper): the paper does NOT specify decode order. Notation lists subtask first (pi_HL(l_{t+1}, m_{t+1} | ...)), Fig 2 draws memory first; no prompt/answer template in main text or appendix. Our subtask-first format is our own choice — order remains a legitimate ablation. The "what was done / should have been done" signal is annotation-time only: the offline LLM gets the subtask sequence + per-subtask success/failure indicator; pi_HL outputs only (subtask, updated memory).

Related finding: the paper is inconsistent about HL conditioning — III-A writes o_t (current obs only, our assumption), III-B writes o_{t-K:t}, Fig 2 draws multiple frames into HL. "HL also sees the video window" is plausible; relevant to Phase 6 (where frame history belongs).

Trade-offs noted:
- Subtask-first enables the "skip summary decode when subtask unchanged" latency optimization; summary-first pays full summary decode every HL call before LL gets its subtask.
- Summary-first puts summary decode in the subtask's causal path (drift propagates); subtask-first insulates action selection.
- Training is indifferent: same CE loss, token order only — swap lives entirely in build_generation_answer/parse_generation_answer.

Status: parked; keeping subtask-first for now. Revisit as an ablation once training runs exist, after re-reading the paper's actual decode order.
