import hashlib, json
from pathlib import Path
import pandas as pd
from gate.policy_c import apply_policy_c, index, snapshot_features, snapshot_labels
from gate.run import main

ROOT=Path(__file__).resolve().parents[1]
def test_policy_c_baseline_equivalence():
 raw={n:pd.read_csv(ROOT/'data'/'synthetic'/'v2'/'raw'/f'{n}.csv',dtype=str,keep_default_na=False) for n in ('productions','budget_revisions','category_plans','commitments','manual_costs','payments','snapshot_schedule')}
 p=apply_policy_c(raw); ix=index({n:p.tables[n] for n in ('productions','budget_revisions','category_plans','commitments','manual_costs')}); rows=[]
 for r in p.eligibility.sort_values(['production_id','stage']).to_dict('records'):
  rows.append({'production_id':r['production_id'],'stage':r['stage'],'as_of_date':r['as_of_date'],'eligible':bool(r['eligible']),'reason':r['reason'],'target_reliable':bool(p.target_reliable[r['production_id']]),'labels':snapshot_labels(ix,r['production_id'],r['as_of_date']),'features':snapshot_features(ix,r['production_id'],r['stage'],r['as_of_date']) if r['eligible'] else None})
 digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest(); fixture=json.loads((ROOT/'tests'/'fixtures'/'policy_c_baseline.json').read_text())
 assert digest==fixture['digest']; assert [{'production_id':x['production_id'],'stage':x['stage'],'reason':x['reason']} for x in rows if not x['eligible']]==fixture['ineligible']
def test_source_gate_is_deterministic(tmp_path):
 a=tmp_path/'a'; b=tmp_path/'b'; raw=ROOT/'data'/'synthetic'/'v2'/'raw'
 assert main(['--stage','source','--raw',str(raw),'--out',str(a)])==0
 assert main(['--stage','source','--raw',str(raw),'--out',str(b)])==0
 assert (a/'gate_result.json').read_bytes()==(b/'gate_result.json').read_bytes()
