#!/usr/bin/env python3
"""ALGO3 eight-heat and future-run research companion (stdlib only).

Use this *offline* on one completed COMP-1.1/2.0 SQLite archive. No HTTP,
no credentials, no access to RIT, no trading, no hindsight in the policy
simulators other than the ex-post labels they are evaluating.

Outputs per-heat tender reports, shadow-tail cohort analysis, option-to-wait
counterfactuals, and a self-contained Markdown memo. IMPORTANT: all reported
P&L is modeled, not real RIT P&L. In the archive no orders were actually sent.

    python algo3_tournament_research_v2.py --db path/to/algo3_competition.sqlite

If --output-dir is omitted, results go to a new sibling research folder.
"""
from __future__ import annotations
import argparse,csv,json,math,sqlite3,statistics,os
from collections import defaultdict
from pathlib import Path
from typing import Any,Dict,Iterable,Mapping,Optional,Sequence

VERSION='ALGO3-RESEARCH-V2.0'
HORIZONS=(.25,.5,1.,2.,3.,5.,10.,15.,20.)

def f(x,default=None):
 try:
  if x is None:return default
  y=float(x)
  return y if math.isfinite(y) else default
 except (ValueError,TypeError):return default

def med(x):
 y=[float(z) for z in x if z is not None and math.isfinite(float(z))]
 return statistics.median(y) if y else None

def mean(x):
 y=[float(z) for z in x if z is not None and math.isfinite(float(z))]
 return statistics.fmean(y) if y else None

def fmt(x, n=2):return 'N/A' if x is None else f'{x:,.{n}f}'

def save_csv(path:Path,rows:Sequence[Mapping[str,Any]]) -> None:
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as fo:
  fields=list(rows[0].keys());w=csv.DictWriter(fo,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def get_triggers(c:sqlite3.Connection,evals:Mapping[tuple,list]) -> list[dict]:
 triggers=[]
 for mono,tid,payload in c.execute("SELECT mono_ns,tender_id,payload_json FROM events WHERE kind='shadow_trigger' ORDER BY mono_ns"):
  obj=json.loads(payload)
  econ=obj.get('evaluation',{}).get('economics',{})
  matches=[]
  for (ri,t_id),row in evals.items():
   if tid!=t_id:continue
   best=min(row,key=lambda x:abs(x['mono_ns']-mono))
   matches.append((abs(best['mono_ns']-mono),ri,best))
  if not matches:continue
  diff,ri,best=min(matches,key=lambda x:x[0])
  if diff>5_000_000_000:continue
  triggers.append({
    'round_index':ri,'tender_id':tid,'trigger_mono_ns':mono,
    'trigger_edge':f(econ.get('edge_per_share')),
    'trigger_profit':f(econ.get('modeled_profit')),
    'quantity':int(econ.get('requested_qty') or 0),
    'action':econ.get('tender_action'),
    'trigger_tick':best['tick'],
    'participant_share':f(obj.get('participant_context',{}).get('participant_share_proxy')),
  })
 return sorted(triggers,key=lambda x:(x['round_index'],x['tender_id']))

def load_db(db:Path):
 c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
 c.row_factory=sqlite3.Row
 names={row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 required={'events','tender_evaluations','shadow_followups'}
 missing=required-names
 if missing:raise RuntimeError(f'Input is not a COMP-1.1+ SQLite archive; missing tables: {sorted(missing)}')
 evals=defaultdict(list)
 for r in c.execute("SELECT round_index,tender_id,mono_ns,tick,source,edge_per_share,modeled_profit,full_coverage,seconds_to_expiry,raw_json FROM tender_evaluations ORDER BY mono_ns"):
  d=dict(r); evals[(r['round_index'],r['tender_id'])].append(d)
 triggers=get_triggers(c,evals)
 follow=[dict(r) for r in c.execute("SELECT round_index,tender_id,horizon_s,elapsed_s,full_edge,tail_10_incremental_pnl,tail_25_incremental_pnl,tail_40_incremental_pnl,route_participant_share FROM shadow_followups ORDER BY round_index,tender_id,horizon_s")]
 c.close()
 return evals,triggers,follow

def first_hit(g:list[dict],threshold:float):
 return next((x for x in g if x['full_coverage'] and x['edge_per_share'] is not None and x['edge_per_share']>=threshold and (x.get('seconds_to_expiry') is None or x['seconds_to_expiry']>=1)),None)

def option_wait(g:list[dict],delay:float=2.0,aggressive_edge=.08,threshold=.05):
 """Strictly chronological hypothetical option-to-wait strategy.

    After first 5c crossing and if current edge under 10c, either:
    * lock when the edge reaches 8c during the two-second waiting window; OR
    * after the window, accept the next observed full-coverage 5c crossing.

    This can miss bids and the displayed book is not guaranteed fillable.
    """
 start=first_hit(g,threshold)
 if start is None:return None
 if start['edge_per_share']>=.10:return start
 anchor=start['mono_ns'];deadline=anchor+int(delay*1e9)
 for row in g:
  if row['mono_ns']<=anchor or not row['full_coverage'] or row['edge_per_share'] is None:continue
  if row.get('seconds_to_expiry') is not None and row['seconds_to_expiry']<1:continue
  if row['mono_ns']<=deadline and row['edge_per_share']>=aggressive_edge:return row
  if row['mono_ns']>=deadline and row['edge_per_share']>=threshold:return row
 return None

def cohort_name(edge:float):return '5-<10c' if edge<.10 else '>=10c'

def generate(db:Path,out:Path) -> dict:
 out.mkdir(parents=True,exist_ok=True)
 evals,triggers,follow=load_db(db)
 trig_by_key={(x['round_index'],x['tender_id']):x for x in triggers}
 tender_rows=[]; option_rows=[]
 for key,g in sorted(evals.items()):
  g.sort(key=lambda x:x['mono_ns'])
  first=g[0]; full=[x for x in g if x['full_coverage'] and x['edge_per_share'] is not None]
  maxrow=max(full,key=lambda x:x['edge_per_share']) if full else None
  trig=trig_by_key.get(key)
  tender_rows.append({
   'round_index':key[0],'tender_id':key[1], 'tick_first':first['tick'],
   'arrival_edge':first['edge_per_share'],
   'max_full_edge':maxrow['edge_per_share'] if maxrow else None,
   'max_tick':maxrow['tick'] if maxrow else None,
   'full_coverage_sample_ratio':len(full)/max(1,len(g)),
   'accepted_by_shadow':int(trig is not None),'trigger_edge':trig['trigger_edge'] if trig else None,
   'trigger_profit':trig['trigger_profit'] if trig else None,
  })
  im=first_hit(g,.05)
  w2=option_wait(g,2.,.08,.05)
  for label,row in [('instant_5c',im),('wait_2s_or_8c',w2),('instant_8c',first_hit(g,.08)),('instant_10c',first_hit(g,.10))]:
   option_rows.append({
    'round_index':key[0],'tender_id':key[1],'policy':label,
    'selected':int(row is not None),
    'tick':row['tick'] if row else None,
    'edge':row['edge_per_share'] if row else None,
    'modeled_profit':row['modeled_profit'] if row else 0.,
    'delay_from_first_5c_s':(row['mono_ns']-im['mono_ns'])/1e9 if row and im else None,
   })
 followup_rows=[]
 for a in follow:
  key=(a['round_index'],a['tender_id']); t=trig_by_key.get(key)
  if t is None or t.get('trigger_edge') is None:continue
  row=dict(a)
  row['trigger_edge']=t['trigger_edge']
  row['entry_cohort']=cohort_name(t['trigger_edge'])
  row['trigger_quantity']=t['quantity']
  row['participant_share_trigger']=t['participant_share']
  row['full_edge_delta']=f(a['full_edge'])-t['trigger_edge'] if f(a['full_edge']) is not None else None
  followup_rows.append(row)
 regime=[]
 for cohort in ('all','5-<10c','>=10c'):
  for h in HORIZONS:
   rows=[x for x in followup_rows if abs(x['horizon_s']-h)<.000001 and (cohort=='all' or x['entry_cohort']==cohort)]
   vals=[f(x.get('tail_25_incremental_pnl')) for x in rows]; vals=[x for x in vals if x is not None]
   regime.append(dict(cohort=cohort,horizon_s=h,n=len(vals),sum_tail25_proxy=sum(vals),mean_tail25_proxy=mean(vals),win_frac=sum(v>0 for v in vals)/len(vals) if vals else None,median_tail25_proxy=med(vals),max_loss_proxy=min(vals) if vals else None))
 rounds=[]
 for ri in sorted({x['round_index'] for x in tender_rows}):
  tt=[x for x in tender_rows if x['round_index']==ri]
  rr=[x for x in triggers if x['round_index']==ri]
  rounds.append({'round_index':ri,'tenders':len(tt),'triggered':len(rr),'trigger_profit_sum':sum(x['trigger_profit'] or 0 for x in rr),'full_coverage_any':sum(x['max_full_edge'] is not None for x in tt)})
 save_csv(out/'tenders.csv',tender_rows)
 save_csv(out/'triggers.csv',triggers)
 save_csv(out/'rounds.csv',rounds)
 save_csv(out/'followups.csv',followup_rows)
 save_csv(out/'tail_regimes.csv',regime)
 save_csv(out/'option_timing_candidates.csv',option_rows)
 total=sum(x['trigger_profit'] or 0 for x in triggers)
 low=[x for x in triggers if x['trigger_edge'] is not None and x['trigger_edge']<.10]
 high=[x for x in triggers if x['trigger_edge'] is not None and x['trigger_edge']>=.10]
 f10={x['cohort']:x for x in regime if x['horizon_s']==10.0}
 policy=[]
 for label in ('instant_5c','wait_2s_or_8c','instant_8c','instant_10c'):
  rr=[x for x in option_rows if x['policy']==label]
  policy.append({'policy':label,'selected':sum(x['selected'] for x in rr), 'total_model_profit':sum(x['modeled_profit'] or 0 for x in rr),
    'train_model_profit':sum(x['modeled_profit'] or 0 for x in rr if x['round_index']<=4),
    'test_model_profit':sum(x['modeled_profit'] or 0 for x in rr if x['round_index']>=5)})
 save_csv(out/'entry_policy_summary.csv',policy)
 md=[f'# ALGO3 COMP-2.0: Eight-Heat Research Memo',
     '',f'Input: `{db.name}`; generated by `{VERSION}`.',
     '',f'**Coverage:** {len(rounds)} rounds, {len(evals)} tenders, {len(triggers)} 5¢ shadow triggers, {len(follow)} logged followups.',
     '',f'**After-fee displayed-book trigger profit (before execution slippage):** ${total:,.2f}, with **zero live trades in shadow**. Do not call this realized P&L.',
     '',f'Entry 5–under-10¢ group: {len(low)} triggers. Entry ≥10¢ group: {len(high)} triggers.',
     '', '## 25% no-own-impact marginal-tail proxy',
     '', '| Hold | All | 5–under-10¢ | ≥10¢ |',
     '|---|---:|---:|---:|']
 for h in (1.,3.,5.,10.,15.,20.):
  r={x['cohort']:x for x in regime if x['horizon_s']==h}
  md.append('| '+f'{h:g}s'+' | '+' | '.join(f'${r[k]["sum_tail25_proxy"]:,.0f} (n={r[k]["n"]})' for k in ('all','5-<10c','>=10c'))+' |')
 md+=['','**Interpretation:** Full-block shadow quotes are not executable net profit: there were no core hedge orders in the source heats, no spread paid in the tail counterfactual, and no dynamic impact/replenishment modeling. Treat differences as labels for hypotheses only.',
     '', 'The eight-heat result **rejects an unconditional 10–20s tail**. The conditional low-entry-edge subgroup is the research lead. The 10¢+ group should ordinarily lock the exceptionally favorable edge immediately.',
     '', '## Tender-option timing (chronological snapshots, same caveats)',
     '', '| Policy | Selected | All-round model | Rounds 1–4 | Rounds 5–8 |', '|---|---:|---:|---:|---:|']
 for x in policy:
  md.append(f'| {x["policy"]} | {x["selected"]} | ${x["total_model_profit"]:,.0f} | ${x["train_model_profit"]:,.0f} | ${x["test_model_profit"]:,.0f} |')
 md+=['','A 2-second wait on 5–under-10¢ triggers can improve modeled prices but **misses opportunities**. The instant threshold remains the live default. Do not optimize simultaneously over many delays and then present the best in-sample policy as validated.',
     '', '## COMP-2.0 decisions',
     '', '- **Live default:** calibrated 100%-immediate-booking-gated SOR burst, unchanged threshold 5¢.',
     '- **Research extension:** conditional core+tail for 5–under-10¢ edges, tested offline and behind two additional live opt-ins.',
     '- **Gates:** core orders must be completely filled/reported and mirrored inventory confirmed before the tail clock starts.',
     '- **Tail exits:** holding deadline, exposure mismatch, fine increase, end-of-round, stale/insufficient book, or configured edge stop/take. Every exit returns through full-order reconciliation.',
     '- **Next prerequisite before relying on live tail:** one controlled demo execution and audit of partial-fill, venue, position, and accounting behavior.',
     '', '## Evidence limitations',
     '', 'Only eight heats from one session were studied; conditions are not independent; participant order behavior changes; fees/tender semantics can change on competition day; mock server tests verify software logic but not live simulator fill kinetics.',
     '']
 (out/'analysis_report.md').write_text('\n'.join(md),encoding='utf-8')
 summary={'version':VERSION,'db':str(db),'rounds':len(rounds),'tenders':len(evals),'triggers':len(triggers),'shadow_followups':len(follow),'model_profit':total,'tail_10s_25pct_by_cohort':{k:{j: v for j,v in x.items() if j!='cohort'} for k,x in f10.items()},'entry_policy_simulations':policy}
 (out/'research_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 return summary

def main():
 p=argparse.ArgumentParser(description='Offline ALGO3 tender/tail research from COMP-1.1/2.0 SQLite')
 p.add_argument('--db',required=True,type=Path,help='completed algo3_competition.sqlite from a shadow run')
 p.add_argument('--output-dir',type=Path,default=None)
 a=p.parse_args()
 if not a.db.is_file():raise SystemExit('Missing input SQLite database: '+str(a.db))
 output=a.output_dir or a.db.parent/(a.db.stem+'_research_v2')
 result=generate(a.db,output)
 print(json.dumps({k:v for k,v in result.items() if k not in ('entry_policy_simulations','tail_10s_25pct_by_cohort')},indent=2))
 print('Report:',output/'analysis_report.md')
 return 0

if __name__=='__main__':raise SystemExit(main())
