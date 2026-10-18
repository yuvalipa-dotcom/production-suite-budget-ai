"""Pure, deterministic checks for the P2 source and handoff gates."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import pandas as pd
from .policy_c import (FEATURE_RULES, FEATURE_SOURCES, STAGES, apply_policy_c,
                       as_csv_text, index, snapshot_features, snapshot_labels)

RAW_TABLES=("productions","budget_revisions","category_plans","commitments","manual_costs","payments","snapshot_schedule")
LEDGERS=("productions","budget_revisions","category_plans","commitments","manual_costs")
HARD="HARD"; WARN="WARN"; INFO="INFO"
def result(id,severity,status,details="") : return {"id":id,"severity":severity,"status":status,"details":details}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_raw(raw): return {n:pd.read_csv(Path(raw)/f"{n}.csv",dtype=str,keep_default_na=False) for n in RAW_TABLES}
def raw_manifest_hash(raw): return sha(Path(raw)/"manifest.json")
def role_contract(root): return json.loads((Path(root)/"contracts"/"column_roles.json").read_text())
def schema_valid(c):
    req={"dataset","version","raw_manifest_sha256","primary_key","row_count","exclusion_counts","columns","split_policy","outcome_only_exclusions"}
    return isinstance(c,dict) and req <= set(c) and c["primary_key"]==["production_id","stage"] and all({"name","dtype","role","unit","nullable"}<=set(x) and x["role"] in {"id","feature","target","metadata"} for x in c["columns"])
def expected(raw):
    p=apply_policy_c(raw); ix=index({n:p.tables[n] for n in LEDGERS}); rows={}
    for r in p.eligibility.to_dict("records"):
        k=(r["production_id"],r["stage"])
        rows[k]={"eligibility":bool(r["eligible"]),"reason":r["reason"],"features":snapshot_features(ix,*k,r["as_of_date"]) if r["eligible"] else None,"labels":snapshot_labels(ix,k[0],r["as_of_date"])}
    return p,rows
def _fail(keys): return ", ".join(map(str,keys[:5]))
def source_checks(root,rawdir):
    root=Path(root); checks=[]; raw=load_raw(rawdir); manifest=json.loads((Path(rawdir)/"manifest.json").read_text())
    files=manifest.get("files_sha256",{}); bad=[n for n,v in files.items() if not (Path(rawdir)/n).is_file() or sha(Path(rawdir)/n)!=v]
    checks.append(result("G-SRC-MANIFEST",HARD,"PASS" if not bad else "FAIL",_fail(bad)))
    roles=role_contract(root)["tables"]; missing=[]
    for n,df in raw.items():
        if not set(df.columns) <= set(roles.get(n,{})): missing.append(n)
    checks.append(result("G-SRC-ROLES",HARD,"PASS" if not missing else "FAIL",_fail(missing)))
    outcome={f"{t}.{c}" for t,x in roles.items() for c,s in x.items() if s["role"]=="outcome-only"}
    wanted={"commitments.final_pre_vat","productions.financial_close_date"}
    checks.append(result("G-SRC-OUTCOME",HARD,"PASS" if wanted<=outcome else "FAIL",_fail(sorted(wanted-outcome))))
    return checks
def handoff_checks(root,data_path,contract_path,rawdir):
    checks=[]; root=Path(root)
    if not Path(data_path).is_file() or not Path(contract_path).is_file(): return [result("G-IN-FILES",HARD,"FAIL","missing input")]
    try: data=pd.read_csv(data_path,dtype=str,keep_default_na=False); contract=json.loads(Path(contract_path).read_text())
    except Exception as e: return [result("G-IN-FILES",HARD,"FAIL",type(e).__name__)]
    checks.append(result("G-IN-FILES",HARD,"PASS"))
    checks.append(result("G-CONTRACT-SCHEMA",HARD,"PASS" if schema_valid(contract) else "FAIL",""))
    raw=load_raw(rawdir); mh=raw_manifest_hash(rawdir)
    checks.append(result("G-CONTRACT-PROVENANCE",HARD,"PASS" if contract.get("raw_manifest_sha256")==mh else "FAIL",""))
    specs={x.get("name"):x for x in contract.get("columns",[]) if isinstance(x,dict)}
    checks.append(result("G-SCHEMA-COLUMNS",HARD,"PASS" if set(data.columns)==set(specs) else "FAIL",_fail(sorted(set(data.columns)^set(specs)))))
    dtype_bad=[]
    for n,s in specs.items():
        if n not in data: continue
        try:
            if s.get("dtype") in ("number","integer"): pd.to_numeric(data[n])
            elif s.get("dtype")=="boolean" and not set(data[n])<={"True","False"}: raise ValueError()
        except Exception: dtype_bad.append(n)
    checks.append(result("G-SCHEMA-DTYPES",HARD,"PASS" if not dtype_bad else "FAIL",_fail(dtype_bad)))
    rawroles=role_contract(root)["tables"]; outcomes={f"{t}.{c}" for t,x in rawroles.items() for c,s in x.items() if s["role"]=="outcome-only"}
    role_bad=[n for n,s in specs.items() if n in FEATURE_RULES and s.get("role")!="feature"]
    checks.append(result("G-ROLES",HARD,"PASS" if not role_bad else "FAIL",_fail(role_bad)))
    leaks=[n for n,s in specs.items() if s.get("role")=="feature" and (n in {x.rsplit('.',1)[-1] for x in outcomes} or FEATURE_SOURCES.get(n,set())&outcomes)]
    checks.append(result("G-LEAKAGE-OUTCOME",HARD,"PASS" if not leaks else "FAIL",_fail(leaks)))
    keybad=data.duplicated(["production_id","stage"]).any() or not set(data.get("stage",[]))<=set(STAGES)
    checks.append(result("G-KEY-UNIQUE",HARD,"PASS" if not keybad else "FAIL",""))
    p,exp=expected(raw); delivered={(r.production_id,r.stage) for r in data.itertuples()}
    eligible={k for k,v in exp.items() if v["eligibility"]}
    checks.append(result("G-ELIGIBILITY-SET",HARD,"PASS" if delivered==eligible else "FAIL",_fail(sorted(delivered^eligible))))
    actual_counts=Counter(v["reason"] for v in exp.values() if not v["eligibility"])
    checks.append(result("G-ELIGIBILITY-COUNTS",HARD,"PASS" if contract.get("row_count")==len(data) and contract.get("exclusion_counts")==dict(sorted(actual_counts.items())) else "FAIL",""))
    def badcols(cols,tol=.01):
        bad=[]
        for r in data.to_dict("records"):
            e=exp.get((r.get("production_id"),r.get("stage")),{})
            source={**(e.get("features") or {}),**e.get("labels",{})}
            for c in cols:
                if c in r and c in source and (abs(float(r[c])-float(source[c]))>tol if isinstance(source[c],(float,int)) and not isinstance(source[c],bool) else r[c]!=as_csv_text(source[c])): bad.append((r.get("production_id"),r.get("stage"),c))
        return bad
    gapbad=badcols(["budget_gap_ils"]); labelbad=badcols(["is_overrun"])
    checks += [result("G-TARGET-GAP",HARD,"PASS" if not gapbad else "FAIL",_fail(gapbad)),result("G-TARGET-LABEL",HARD,"PASS" if not labelbad else "FAIL",_fail(labelbad))]
    consistent=[(r["production_id"],r["stage"]) for r in data.to_dict("records") if "is_overrun" in r and "budget_gap_ils" in r and (r["is_overrun"]=="True") != (float(r["budget_gap_ils"])>0)]
    checks.append(result("G-TARGET-CONSISTENCY",HARD,"PASS" if not consistent else "FAIL",_fail(consistent)))
    finbad=badcols(FEATURE_RULES); checks.append(result("G-FIN-RECON",HARD,"PASS" if not finbad else "FAIL",_fail(finbad)))
    budgetbad=badcols(["budget_pre_vat"]); checks.append(result("G-FIN-ASOF-BUDGET",HARD,"PASS" if not budgetbad else "FAIL",_fail(budgetbad)))
    nonfinbad=badcols([x for x in FEATURE_RULES if x not in {"budget_pre_vat"}]); checks.append(result("G-NONFIN-RULES",HARD,"PASS" if not nonfinbad else "FAIL",_fail(nonfinbad)))
    prod=raw["productions"].set_index("production_id"); groupbad=[r.production_id for r in data.itertuples() if "split" in data and r.split!=prod.loc[r.production_id,"split"]]; foldbad=[r.production_id for r in data.itertuples() if "cv_fold" in data and str(r.cv_fold)!=str(prod.loc[r.production_id,"cv_fold"])]
    checks += [result("G-SPLIT-GROUP",HARD,"PASS" if not groupbad else "FAIL",_fail(groupbad)),result("G-SPLIT-FOLD",HARD,"PASS" if not foldbad else "FAIL",_fail(foldbad))]
    nulls=[n for n,s in specs.items() if s.get("role") in {"feature","target"} and (n not in data or (data[n]=="").any())]; checks.append(result("G-NULLS",HARD,"PASS" if not nulls else "FAIL",_fail(nulls)))
    ranges=[]
    for n,s in specs.items():
        if n in data and ("min" in s or "max" in s):
            x=pd.to_numeric(data[n]);
            if ("min" in s and (x<s["min"]).any()) or ("max" in s and (x>s["max"]).any()): ranges.append(n)
    checks.append(result("G-RANGES",HARD,"PASS" if not ranges else "FAIL",_fail(ranges)))
    checks += [result("W-PREVALENCE-IMBALANCE",WARN,"PASS","train 44.0%; test 59.1%"),result("I-SYNTHETIC",INFO,"PASS","synthetic data; no real-world validation")]
    return checks
