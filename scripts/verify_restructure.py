"""
One-shot verification that the folder restructure didn't break anything.
Runs Tier 1 items 2, 3, 4 + key Tier 2 cross-checks. Prints PASS/FAIL per check.

Usage:
    python scripts/verify_restructure.py
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

results = []

def check(name, ok, detail=''):
    tag = f'{GREEN}PASS{RESET}' if ok else f'{RED}FAIL{RESET}'
    print(f'[{tag}] {name}' + (f'  -- {detail}' if detail else ''))
    results.append((name, ok, detail))

print(f'\n{YELLOW}=== Tier 1.2  Saved model files exist at new paths ==={RESET}')
from app.config import MODEL_FILES, STATS_FILES
for key, path in MODEL_FILES.items():
    check(f'MODEL_FILES[{key}]', os.path.exists(path), path if not os.path.exists(path) else '')
for key, path in STATS_FILES.items():
    check(f'STATS_FILES[{key}]', os.path.exists(path), path if not os.path.exists(path) else '')

print(f'\n{YELLOW}=== Tier 1.2b  No stale flat-layout model paths in app/ ==={RESET}')
stale_patterns = [
    re.compile(r"models[/\\]saved[/\\](?!01_|02_|03_|supplementary)[a-z_]+\.(joblib|pt|keras|json)"),
]
hits = []
for p in (ROOT / 'app').rglob('*.py'):
    text = p.read_text(encoding='utf-8', errors='ignore')
    for pat in stale_patterns:
        for m in pat.finditer(text):
            hits.append(f'{p.relative_to(ROOT)}: {m.group(0)}')
check('No stale model paths in app/*.py', len(hits) == 0, '; '.join(hits[:3]))

print(f'\n{YELLOW}=== Tier 1.2c  ModelManager loads all 5 models without exception ==={RESET}')
try:
    from app.models.model_manager import ModelManager
    mm = ModelManager()  # __init__ calls _load_all
    available = list(mm.models.keys())
    check('ModelManager init + _load_all()', True, f'loaded: {available}')
    expected = {'XGBoost (Class Weights)', 'XGBoost (SMOTE+Tuned)', 'AE+XGBoost', 'AE+BDS+XGBoost', 'LSTM+RF'}
    missing = expected - set(available)
    check('All 5 models registered', not missing, f'missing: {missing}' if missing else '')
except Exception as e:
    check('ModelManager init + _load_all()', False, f'{type(e).__name__}: {e}')

print(f'\n{YELLOW}=== Tier 1.3  verified_metrics.json sanity ==={RESET}')
vm_path = ROOT / 'results' / 'verified_metrics.json'
check('verified_metrics.json exists', vm_path.exists())
if vm_path.exists():
    vm = json.loads(vm_path.read_text())
    # Check F1 at threshold 0.5 for each model name substring
    expected_f1 = {
        'XGBoost Baseline (CW)':   0.5215,
        'XGBoost SMOTE+tuned':     0.8646,
        'AE + XGBoost':            0.8690,
        'AE + BDS + XGBoost':      0.8706,
    }
    rows = vm.get('models', [])
    for name_sub, target in expected_f1.items():
        match = next((r for r in rows
                      if r.get('threshold') == 0.5 and r.get('model', '').startswith(name_sub)), None)
        if match is None:
            check(f'F1 ~{target:.4f} for "{name_sub}"', False, 'no row found')
        else:
            ok = abs(match['f1'] - target) < 5e-4
            check(f'F1 ~{target:.4f} for "{name_sub}"', ok,
                  f'got {match["f1"]:.4f}')

print(f'\n{YELLOW}=== Tier 2.5  No dead route refs to removed templates ==={RESET}')
dead = ['welcome', 'batch', 'alerts']
for word in dead:
    pat = re.compile(rf"url_for\(\s*['\"][a-zA-Z._]*{word}")
    hits = []
    for p in (ROOT / 'app' / 'templates').rglob('*.html'):
        for m in pat.finditer(p.read_text(encoding='utf-8', errors='ignore')):
            hits.append(f'{p.name}:{m.group(0)}')
    check(f'No url_for to "{word}" in templates', len(hits) == 0, '; '.join(hits[:3]))

# report_generator import
hits = []
for p in (ROOT / 'app').rglob('*.py'):
    text = p.read_text(encoding='utf-8', errors='ignore')
    if re.search(r'\bimport\s+report_generator|\bfrom\s+\S*report_generator', text):
        hits.append(str(p.relative_to(ROOT)))
check('No live import of removed report_generator', len(hits) == 0, '; '.join(hits))

print(f'\n{YELLOW}=== Tier 2.6  Templates reflect 3-model framing ==={RESET}')
forbidden = ['Staged 4-Model', 'proposed_component', '(Modified)', '(Component)']
for word in forbidden:
    hits = []
    for p in (ROOT / 'app' / 'templates').rglob('*.html'):
        if word in p.read_text(encoding='utf-8', errors='ignore'):
            hits.append(p.name)
    check(f'No occurrences of "{word}" in templates', len(hits) == 0, '; '.join(hits))

print(f'\n{YELLOW}=== Tier 2.7  config.py number consistency ==={RESET}')
from app.config import MODEL_F1_LABELS, STAGED_STUDY_TABLE_A, STAGED_STUDY_TABLE_B
expected = {
    'xgboost_baseline': '0.5215',
    'xgboost_smote':    '0.8646',
    'ae_xgboost':       '0.8690',
    'ae_bds_xgboost':   '0.8706',
    'lstm_rf_hybrid':   '0.7892',
}
for code, f1str in expected.items():
    check(f'MODEL_F1_LABELS[{code}] = {f1str}', f1str in MODEL_F1_LABELS.get(code, ''))

table_a_f1s = sorted([r['f1'] for r in STAGED_STUDY_TABLE_A])
check('Table A has 3 rows', len(STAGED_STUDY_TABLE_A) == 3, f'got {len(STAGED_STUDY_TABLE_A)}')
check('Table A F1s match study (0.7892, 0.8646, 0.8706)',
      table_a_f1s == [0.7892, 0.8646, 0.8706],
      f'got {table_a_f1s}')

table_b_f1s = [r['f1'] for r in STAGED_STUDY_TABLE_B]
check('Table B is monotonic ascending (ablation ladder)',
      table_b_f1s == sorted(table_b_f1s),
      f'got {table_b_f1s}')

print(f'\n{YELLOW}=== Summary ==={RESET}')
passed = sum(1 for _, ok, _ in results if ok)
failed = len(results) - passed
print(f'{GREEN}{passed} passed{RESET}, {RED if failed else GREEN}{failed} failed{RESET}, total {len(results)}')
sys.exit(0 if failed == 0 else 1)
