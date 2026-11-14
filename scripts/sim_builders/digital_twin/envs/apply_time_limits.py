"""Per-tier episode time limits for the twin eval (user 2026-10-30: L1 30 s, L2 45 s, L3 90 s, L4 120 s).
Adds TIER_TIME_S to tasks.py and makes Task.overrides() emit --env.processor.reset.control_time_s for twin tiers (SIM_EVAL_ARGS on the
command line still comes later and wins). Apply ONLY between eval runs: every layout re-reads tasks.py when it launches.
usage (LeRobot root): python3 lerobot/scripts/sim_builders/digital_twin/envs/apply_time_limits.py [--check]"""
import pathlib, sys
p = pathlib.Path("lerobot/src/lerobot/robots/rebot_b601_sim/tasks.py"); s = p.read_text()
old_a = "TWIN_TIERS = TIERS[2:]\n"
new_a = old_a + '''TIER_TIME_S = {"twin_l1": 30.0, "twin_l2": 45.0, "twin_l3": 90.0, "twin_l4": 120.0}  # episode budget per twin tier (user 2026-10-30);
# res600 run at 60 s: L1 successes finished by 25 s, L2 by 50 s (one), L4 sequences cannot finish in 60 s. Old tiers keep the config's value.
'''
old_b = '''            f"--policy.eval_subtask_contacts={json.dumps(list(self.eval_subtask_contacts))}",
        ]
'''
new_b = '''            f"--policy.eval_subtask_contacts={json.dumps(list(self.eval_subtask_contacts))}",
        ] + ([f"--env.processor.reset.control_time_s={TIER_TIME_S[self.tier]}"] if self.tier in TIER_TIME_S else [])
'''
ok = s.count(old_a) == 1 and s.count(old_b) == 1 and "TIER_TIME_S" not in s
if "--check" in sys.argv: print("applies cleanly" if ok else "does NOT apply (already applied or text changed)"); sys.exit(0 if ok else 1)
assert ok, "patch does not apply"
p.write_text(s.replace(old_a, new_a).replace(old_b, new_b)); print("applied: TIER_TIME_S + overrides() in", p)
