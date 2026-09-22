"""Require a nonempty, valid report for each enabled probe (no model imports)."""
import ast
import json
import sys
from pathlib import Path
import yaml

config, reports = map(Path, sys.argv[1:])
parameters = yaml.safe_load(config.read_text())['probe_parameters']
tree = ast.parse(Path('lerobot/src/lerobot/scripts/rl_offline.py').read_text())
registry = {
    call.args[0].value: call.args[2].value
    for node in tree.body if isinstance(node, ast.Assign)
    if any(isinstance(t, ast.Name) and t.id == '_VALIDATION_PROBES' for t in node.targets)
    for call in node.value.elts
}
enabled = [key for key, value in parameters.items() if key.startswith('enable_') and value]
assert enabled, 'No probes enabled'
failed=[]
for flag in enabled:
    name=registry.get(flag)
    try:
        assert name, f'Unregistered flag: {flag}'
        report=json.loads((reports/name/'index.json').read_text())
        assert report.get('id') == name, f'Unexpected report identity: {report.get("id")}'
    except (OSError, ValueError, AssertionError) as exc:
        failed.append(f'{flag}: {exc}')
if failed:
    raise SystemExit('Incomplete probe suite:\n'+'\n'.join(failed))
print(f'All {len(enabled)} enabled probe reports complete: {reports}')
