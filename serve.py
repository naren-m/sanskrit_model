"""serve.py — local web UI for the Sanskrit engine. Stdlib only (no Flask).

  uv run serve.py            # -> http://127.0.0.1:8008
  uv run serve.py --port 9000

Loads the trained model once and exposes /api/<task>. The page shows, for every
query, the model's proposal beside Panini's verdict.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from slm.infer import Inference

ENGINE: Inference | None = None

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Sanskrit Engine — neural proposes, Pāṇini disposes</title>
<style>
:root{--bg:#0e1116;--card:#171c26;--edge:#232a37;--fg:#e6edf3;--dim:#8b98a9;
--ok:#3fb950;--no:#f85149;--acc:#d29922;--blue:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:28px 20px 8px;text-align:center}
h1{margin:0;font-size:22px;font-weight:650}
.sub{color:var(--dim);font-size:13px;margin-top:4px}
.wrap{max-width:820px;margin:0 auto;padding:16px 16px 60px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:18px 0 10px}
.tab{padding:7px 14px;border:1px solid var(--edge);border-radius:8px;cursor:pointer;
background:var(--card);color:var(--dim);font-size:13px}
.tab.on{color:var(--fg);border-color:var(--blue);background:#1b2432}
.card{background:var(--card);border:1px solid var(--edge);border-radius:12px;
padding:18px;margin-top:12px}
.row{display:flex;gap:8px;flex-wrap:wrap}
input{flex:1;min-width:120px;padding:10px 12px;border:1px solid var(--edge);
border-radius:8px;background:#0d1117;color:var(--fg);font:14px monospace}
button{padding:10px 18px;border:0;border-radius:8px;background:var(--blue);
color:#001;font-weight:650;cursor:pointer}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
.out{margin-top:14px;font-family:ui-monospace,monospace;font-size:14px}
.prop{color:var(--blue)}.ok{color:var(--ok)}.no{color:var(--no)}
.mut{color:var(--dim)}.big{font-size:18px;font-weight:650}
table{border-collapse:collapse;margin-top:8px;width:100%}
td{padding:3px 10px 3px 0;color:var(--dim);font-size:13px;font-family:monospace}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:12px;
border:1px solid var(--edge)}.pill.ok{border-color:var(--ok)}.pill.no{border-color:var(--no)}
.pada{display:inline-block;background:#1b2432;border:1px solid var(--edge);
border-radius:6px;padding:3px 9px;margin:3px 4px 0 0;font-family:monospace}
.wt{letter-spacing:3px;font-family:monospace}
</style></head><body>
<header><h1>Sanskrit Engine</h1>
<div class=sub>neural proposes · <b>Pāṇini disposes</b> &nbsp; <span id=bpb class=mut></span></div>
</header>
<div class=wrap>
 <div class=tabs id=tabs>
  <div class=tab data-t=morph>morphology</div>
  <div class=tab data-t=sandhi>sandhi</div>
  <div class=tab data-t=seg>segmentation</div>
  <div class=tab data-t=meter>chandas</div>
 </div>
 <div class=card>
  <div class=row id=inputs></div>
  <div class=hint id=hint></div>
  <div class=out id=out></div>
 </div>
</div>
<script>
const TASKS={
 morph:{f:["root"],ph:["gam / BU / kf / vad …"],hint:"Enter a bare SLP1 root. The net proposes a Dhātupāṭha analysis; the engine verifies it exists."},
 sandhi:{f:["a","b"],ph:["rAma","asti"],hint:"Two padas. Net joins them; the rule engine gives the licensed sandhi."},
 seg:{f:["text"],ph:["rAmo'sti / sUryodayaH"],hint:"Continuous SLP1. Net splits it into padas."},
 meter:{f:["line"],ph:["vAgarTAviva / rAmAya"],hint:"An SLP1 pāda. Engine scans laghu/guru; net names the meter."},
};
let cur="morph";
const tabs=document.getElementById("tabs"),inp=document.getElementById("inputs"),
 hint=document.getElementById("hint"),out=document.getElementById("out");
function draw(){
 [...tabs.children].forEach(c=>c.classList.toggle("on",c.dataset.t===cur));
 const t=TASKS[cur];inp.innerHTML="";
 t.f.forEach((name,i)=>{const e=document.createElement("input");
  e.id="f"+i;e.placeholder=t.ph[i]||name;e.value=(t.ph[i]||"").split(" / ")[0].split(" ")[0];
  e.onkeydown=ev=>{if(ev.key==="Enter")go()};inp.appendChild(e);});
 const b=document.createElement("button");b.textContent="Analyze";b.onclick=go;inp.appendChild(b);
 hint.textContent=t.hint;out.innerHTML="";
}
tabs.onclick=e=>{if(e.target.dataset.t){cur=e.target.dataset.t;draw();go();}};
async function go(){
 const t=TASKS[cur];const q=t.f.map((_,i)=>"a"+i+"="+encodeURIComponent(document.getElementById("f"+i).value.trim()));
 out.innerHTML="<span class=mut>…</span>";
 const r=await fetch("/api/"+cur+"?"+q.join("&"));const d=await r.json();render(d);
}
function render(d){
 if(d.error){out.innerHTML="<span class=no>"+d.error+"</span>";return}
 if(d.task==="morph"){
  const p=d.proposal;let h=`<div><span class=prop>model proposes</span> → dhātu <b>${p.dhatu||"?"}</b>, gaṇa ${p.gana||"?"}, artha ‘${p.artha||"?"}’</div>`;
  h+=d.verified?`<div class=big><span class=pill ok>✓ Pāṇini confirms</span> ‘${d.clean_root}’ — ${d.entries.length} entr${d.entries.length==1?"y":"ies"}</div>`
              :`<div class=big><span class=pill no>✗ rejected</span> no Pāṇinian derivation for ‘${d.clean_root}’</div>`;
  if(d.entries.length){h+="<table>";for(const e of d.entries)h+=`<tr><td>${e.code}</td><td>gaṇa ${e.gana} (${e.gana_name})</td><td>${e.dhatu}</td><td>‘${e.artha}’</td></tr>`;h+="</table>";}
  out.innerHTML=h;
 }else if(d.task==="sandhi"){
  out.innerHTML=`<div><span class=prop>model</span>: ${d.input[0]} + ${d.input[1]} → <b>${d.model}</b></div>`+
   `<div class=mut>rule (${d.category}): → ${d.rule} &nbsp; <span class="pill ${d.match?"ok":"no"}">${d.match?"match":"differs"}</span></div>`;
 }else if(d.task==="seg"){
  out.innerHTML=`<div class=prop>model segments</div><div>${d.padas.map(p=>`<span class=pada>${p}</span>`).join("")||d.model}</div>`;
 }else if(d.task==="meter"){
  out.innerHTML=`<div><span class=prop>chandas</span>: ${d.input}</div>`+
   `<div class=wt>${d.weights}</div><div class=mut>${d.syllables} syllables</div>`+
   `<div>engine best: <b>${d.symbolic_best.name||"?"}</b> <span class=mut>(distance ${d.symbolic_best.distance})</span></div>`+
   `<div class=mut>model guess: ${d.model_name}</div>`;
 }
}
draw();go();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/meta":
            return self._send(200, json.dumps({"val_bpb": ENGINE.val_bpb}))
        if u.path.startswith("/api/"):
            task = u.path[len("/api/"):]
            q = parse_qs(u.query)
            a = [q.get(f"a{i}", [""])[0] for i in range(3)]
            try:
                if task == "morph":
                    res = ENGINE.morph(a[0])
                elif task == "sandhi":
                    res = ENGINE.sandhi_join(a[0], a[1])
                elif task == "seg":
                    res = ENGINE.seg(a[0])
                elif task == "meter":
                    res = ENGINE.meter(a[0])
                else:
                    res = {"error": f"unknown task {task}"}
            except Exception as e:
                res = {"error": str(e)}
            return self._send(200, json.dumps(res, ensure_ascii=False))
        self._send(404, json.dumps({"error": "not found"}))


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()
    print("loading model…")
    ENGINE = Inference()
    print(f"val_bpb={ENGINE.val_bpb:.4f}  device={ENGINE.device}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Sanskrit Engine → http://127.0.0.1:{args.port}\n  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
