from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


def update_dashboard(
    output: Path, snapshot: dict[str, Any], *, append_history: bool = True
) -> None:
    """Persist a snapshot and rewrite a self-contained auto-refreshing dashboard."""
    history_path = output / "dashboard_history.jsonl"
    if append_history:
        with history_path.open("a") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
    history = [
        json.loads(line) for line in history_path.read_text().splitlines() if line
    ]
    data = json.dumps(history, separators=(",", ":")).replace("</", "<\\/")
    latest = html.escape(json.dumps(snapshot, indent=2, sort_keys=True))
    plot_images = "".join(
        f'<section><div class="label">{html.escape(path.stem.replace("_", " ").title())}</div>'
        f'<img class="runplot" alt="{html.escape(path.stem)} plot with labeled axes" '
        f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"/></section>'
        for path in sorted(output.glob("*.png"))
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5"><title>{html.escape(str(snapshot.get("run_id", "Experiment")))}</title>
<style>
:root{{color-scheme:light dark;--bg:#0b1020;--panel:#151c31;--text:#edf2ff;--muted:#9aa8c7;--grid:#2a3553;--a:#67e8f9;--b:#c4b5fd;--good:#86efac;--bad:#fca5a5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px ui-sans-serif,system-ui;padding:24px}}main{{max-width:1180px;margin:auto}}h1{{font-size:20px;margin:0 0 4px}}.sub{{color:var(--muted);margin-bottom:18px}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.stat,section{{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:14px}}.label{{color:var(--muted);font-size:12px}}.value{{font-size:22px;margin-top:5px}}section{{margin-top:12px}}svg{{width:100%;height:300px;display:block}}.runplot{{display:block;width:100%;height:auto;margin-top:10px;border-radius:6px}}.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 0 48px;color:var(--muted);font-size:12px}}.swatch{{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}}pre{{white-space:pre-wrap;word-break:break-word;color:var(--muted);max-height:360px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:7px;text-align:right;border-bottom:1px solid var(--grid)}}th:first-child,td:first-child{{text-align:left}}@media(max-width:600px){{body{{padding:12px}}}}
</style></head><body><main><h1>{html.escape(str(snapshot.get("run_id", "Experiment")))}</h1><div class="sub">Auto-refreshes every 5 seconds · archived with run artifacts</div>
<div class="stats" id="stats"></div><section><div class="label">Progress and ASR history</div><svg id="chart" viewBox="0 0 1000 300" role="img" aria-label="Experiment metrics by checkpoint; rate or fraction from zero to one"></svg><div class="legend"><span><i class="swatch" style="background:var(--a)"></i>Progress</span><span><i class="swatch" style="background:var(--good)"></i>Direct ASR</span><span><i class="swatch" style="background:var(--b)"></i>Baseline ASR</span></div></section>
<section id="objectiveSection"><div class="label">Optimization metrics</div><svg id="objectiveChart" viewBox="0 0 1000 300" role="img" aria-label="Optimization metric values by checkpoint"></svg><div class="legend" id="objectiveLegend"></div></section>
{plot_images}<section><div class="label">Latest metrics</div><div id="metrics"></div></section><section><div class="label">Latest structured progress</div><pre>{latest}</pre></section>
<script>const H={data};const L=H[H.length-1]||{{}};const pct=(x)=>x==null?'—':(100*x).toFixed(1)+'%';const num=(x,d=3)=>x==null?'—':Number(x).toFixed(d);
const eta=L.eta_seconds==null?'—':Math.ceil(L.eta_seconds/60)+' min';const stats=[['Phase',L.phase],['Progress',(L.completed??0)+' / '+(L.total??'—')],['Throughput',num(L.throughput_per_second)+' /s'],['ETA',eta],['Baseline ASR',pct(L.baseline_asr)],['Direct ASR',pct(L.direct_asr)],['Errors',L.error_count??0],['Retries',L.retry_count??0]];
document.getElementById('stats').innerHTML=stats.map(x=>`<div class="stat"><div class="label">${{x[0]}}</div><div class="value">${{x[1]??'—'}}</div></div>`).join('');
const excluded=new Set(['run_id','config_fingerprint','phase','completed','total','completed_fraction','elapsed_seconds','eta_seconds']);const keys=Object.keys(L).filter(k=>!excluded.has(k)&&['number','boolean','string'].includes(typeof L[k])).sort();document.getElementById('metrics').innerHTML='<table><tbody>'+keys.map(k=>`<tr><td>${{k}}</td><td>${{typeof L[k]==='number'?num(L[k],5):L[k]}}</td></tr>`).join('')+'</tbody></table>';
const svg=document.getElementById('chart'),W=1000,Ht=300,left=70,right=24,plotTop=18,bottom=54,n=Math.max(1,H.length-1),x=i=>left+i/n*(W-left-right),y=v=>plotTop+(1-(v??0))*(Ht-plotTop-bottom),xy=(i,v)=>[x(i),y(v)];let s='';
[0,.25,.5,.75,1].forEach(v=>{{s+=`<path d="M${{left}} ${{y(v)}}H${{W-right}}" fill="none" stroke="var(--grid)"/><text x="${{left-10}}" y="${{y(v)+4}}" text-anchor="end" fill="var(--muted)" font-size="12">${{v.toFixed(2)}}</text>`;}});
s+=`<path d="M${{left}} ${{plotTop}}V${{Ht-bottom}}H${{W-right}}" fill="none" stroke="var(--muted)"/><text x="${{(left+W-right)/2}}" y="${{Ht-9}}" text-anchor="middle" fill="var(--muted)" font-size="13">Checkpoint</text><text x="16" y="${{(plotTop+Ht-bottom)/2}}" text-anchor="middle" fill="var(--muted)" font-size="13" transform="rotate(-90 16 ${{(plotTop+Ht-bottom)/2}})">Rate / fraction (0–1)</text><text x="${{left}}" y="${{Ht-bottom+20}}" text-anchor="middle" fill="var(--muted)" font-size="12">1</text><text x="${{W-right}}" y="${{Ht-bottom+20}}" text-anchor="middle" fill="var(--muted)" font-size="12">${{H.length}}</text>`;
[['completed_fraction','var(--a)'],['direct_asr','var(--good)'],['baseline_asr','var(--b)']].forEach(([k,c])=>{{const pts=H.map((r,i)=>xy(i,k==='completed_fraction'?(r.completed||0)/Math.max(1,r.total||1):r[k])).map(q=>q.join(',')).join(' ');s+=`<polyline points="${{pts}}" fill="none" stroke="${{c}}" stroke-width="3"/>`;}});svg.innerHTML=s;
const objectiveKeys=['trajectory_loss','validation_loss','normalized_mse','cosine_distance','forward_kl'].filter(k=>H.some(r=>Number.isFinite(r[k])));const objectiveSection=document.getElementById('objectiveSection');if(!objectiveKeys.length){{objectiveSection.style.display='none';}}else{{const colors=['var(--a)','var(--good)','var(--b)','#fbbf24','#fb7185'],values=objectiveKeys.flatMap(k=>H.map(r=>r[k]).filter(Number.isFinite)),maxValue=Math.max(...values,1e-9),metricY=v=>plotTop+(1-v/maxValue)*(Ht-plotTop-bottom);let os='';[0,.25,.5,.75,1].forEach(frac=>{{const yy=metricY(frac*maxValue);os+=`<path d="M${{left}} ${{yy}}H${{W-right}}" fill="none" stroke="var(--grid)"/><text x="${{left-10}}" y="${{yy+4}}" text-anchor="end" fill="var(--muted)" font-size="12">${{(frac*maxValue).toPrecision(3)}}</text>`;}});os+=`<path d="M${{left}} ${{plotTop}}V${{Ht-bottom}}H${{W-right}}" fill="none" stroke="var(--muted)"/><text x="${{(left+W-right)/2}}" y="${{Ht-9}}" text-anchor="middle" fill="var(--muted)" font-size="13">Checkpoint</text><text x="16" y="${{(plotTop+Ht-bottom)/2}}" text-anchor="middle" fill="var(--muted)" font-size="13" transform="rotate(-90 16 ${{(plotTop+Ht-bottom)/2}})">Metric value</text>`;objectiveKeys.forEach((k,j)=>{{const pts=H.map((r,i)=>Number.isFinite(r[k])?`${{x(i)}},${{metricY(r[k])}}`:null).filter(Boolean).join(' ');os+=`<polyline points="${{pts}}" fill="none" stroke="${{colors[j]}}" stroke-width="3"/>`;}});document.getElementById('objectiveChart').innerHTML=os;document.getElementById('objectiveLegend').innerHTML=objectiveKeys.map((k,j)=>`<span><i class="swatch" style="background:${{colors[j]}}"></i>${{k}}</span>`).join('');}}
</script></main></body></html>"""
    (output / "dashboard.html").write_text(document)
