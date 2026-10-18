"""Command-line entry point for deterministic P2 checks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .checks import handoff_checks, source_checks, sha
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('source','handoff'),required=True); p.add_argument('--raw',required=True); p.add_argument('--out',required=True); p.add_argument('--data'); p.add_argument('--contract'); a=p.parse_args(argv)
 if a.stage=='handoff' and (not a.data or not a.contract): p.error('--data and --contract are required for handoff')
 root=Path(__file__).resolve().parents[1]; checks=source_checks(root,a.raw) if a.stage=='source' else handoff_checks(root,a.data,a.contract,a.raw)
 inputs={'raw_manifest.json':sha(Path(a.raw)/'manifest.json')}
 if a.stage=='handoff': inputs|={'data.csv':sha(a.data),'dataset_contract.json':sha(a.contract)}
 overall='PASS' if all(c['status']=='PASS' or c['severity']!='HARD' for c in checks) else 'FAIL'; obj={'stage':a.stage,'status':overall,'input_sha256s':inputs,'checks':checks,'limitations':['Offline deterministic validation; synthetic data only.']}
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'gate_result.json').write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
 lines=[f'# Quality gate: {a.stage}','',f'**Status:** {overall}','', '| ID | Severity | Status | Details |','|---|---|---|---|']+[f"| {c['id']} | {c['severity']} | {c['status']} | {c['details']} |" for c in checks]+['','## Limitations','- Offline deterministic validation; synthetic data only.','']; (out/'gate_report.md').write_text('\n'.join(lines)); return 0 if overall=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
