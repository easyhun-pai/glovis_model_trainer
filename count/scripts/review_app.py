"""COUNT 라벨 검수 웹앱 (로컬 전용).

사용:  python count/scripts/review_app.py [--port 8765]  →  http://127.0.0.1:8765

입력/저장: data/COUNT/gt_draft/<영상>.csv       통과 이벤트 (클릭 즉시 저장, pred 열에 최초 예측 보존)
           data/COUNT/gt_draft/<영상>_miss.csv  미탐 후보
이미지:    runs/count/<영상>/crops, plate_evidence, runs/count/review/<영상>/miss_ctx
홀드아웃(파일명에 '검증용') 영상은 읽지 않는다.
"""
import argparse, glob, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "data/COUNT/gt_draft"
LABELS = ("cargo", "safety", "work", "invalid")
PAGE = 48
lock = threading.Lock()
DB = {}  # video -> {"event": df, "miss": df|None, "img": {(kind, id): [path, ...]}}


def load():
    for p in sorted(DRAFT.glob("*.csv")):
        v = p.stem
        if v.endswith("_miss") or "검증용" in v: continue
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "pred" not in df: df.insert(df.columns.get_loc("label"), "pred", df.label)
        run = ROOT / "runs/count" / v; img = {}
        chains = pd.read_csv(run / "events.csv", dtype=str).set_index("event").chain.to_dict()
        for e in df.event:
            c = []
            for t in chains[e].split("|"): c += glob.glob(str(run / "crops" / str(int(t) % 1000000) / "*.jpg"))
            c = sorted(c, key=os.path.getsize, reverse=True)[:5]
            img[("event", e)] = c + glob.glob(str(run / "plate_evidence" / f"E{e}_*.jpg"))
        mp = DRAFT / f"{v}_miss.csv"; miss = None
        if mp.exists():
            miss = pd.read_csv(mp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if "pred" not in miss: miss.insert(miss.columns.get_loc("label"), "pred", miss.label)
            for t in miss.event:
                img[("miss", t)] = [str(ROOT / "runs/count/review" / v / "miss_ctx" / f"{t}_{k}.jpg") for k in ("ctx", "crop")]
        DB[v] = dict(event=df, miss=miss, img=img)


def save(v, kind):
    df = DB[v][kind]; p = DRAFT / (f"{v}_miss.csv" if kind == "miss" else f"{v}.csv"); tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig"); os.replace(tmp, p)


def rows(v, group):
    return DB[v]["miss"] if group == "miss" else DB[v]["event"][DB[v]["event"].pred == group]


def state():
    out = []
    for v, d in DB.items():
        e = d["event"]; m = d["miss"]; g = {}
        for k in ("cargo", "safety", "work"):
            x = e[e.pred == k]; g[k] = dict(total=len(x), done=int((x.reviewed == "1").sum()))
        g["miss"] = dict(total=0 if m is None else len(m), done=0 if m is None else int((m.reviewed == "1").sum()))
        out.append(dict(video=v, groups=g, changed=int((e.label != e.pred).sum())))
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype="application/json; charset=utf-8", code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def js(self, o, code=200): self.send(json.dumps(o, ensure_ascii=False).encode(), code=code)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/": return self.send(HTML.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state": return self.js(state())
        if u.path == "/api/items":
            v, g = q["video"], q["group"]; kind = "miss" if g == "miss" else "event"
            with lock:
                df = rows(v, g)
                if q.get("todo") == "1": df = df[df.reviewed != "1"]
                total = len(df); pg = df.iloc[int(q.get("page", 0)) * PAGE:(int(q.get("page", 0)) + 1) * PAGE]
                items = [dict(r, kind=kind, n=len(DB[v]["img"].get((kind, r["event"]), []))) for r in pg.to_dict("records")]
            return self.js(dict(items=items, total=total, page_size=PAGE))
        if u.path.startswith("/img/"):
            _, _, v, kind, i, n = u.path.split("/")
            try: return self.send(Path(DB[v]["img"][(kind, i)][int(n)]).read_bytes(), "image/jpeg")
            except (KeyError, IndexError, OSError): return self.send(b"", "text/plain", 404)
        self.send(b"", "text/plain", 404)

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); v, kind = b["video"], b["kind"]
        if v not in DB or kind not in ("event", "miss"): return self.js(dict(error="잘못된 요청"), 400)
        df = DB[v][kind]
        try:
            with lock:
                if self.path == "/api/label":
                    if b["label"] not in LABELS: return self.js(dict(error="잘못된 라벨"), 400)
                    df.loc[df.event == str(b["id"]), ["label", "reviewed"]] = [b["label"], "1"]
                elif self.path == "/api/confirm":
                    ids = [str(x) for x in b["ids"]]; m = df.event.isin(ids) & (df.label != "")
                    df.loc[m, "reviewed"] = "1"
                else: return self.js(dict(error="없는 경로"), 404)
                save(v, kind)
        except PermissionError:
            return self.js(dict(error="CSV 가 다른 프로그램에서 열려 있어 저장하지 못했습니다. 닫고 다시 눌러주세요."), 409)
        self.js(dict(ok=True))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>COUNT 라벨 검수</title><style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1c2230;--mute:#667085;--line:#d9dee7;--accent:#2457d6;--cargo:#14804a;--safety:#c2410c;--work:#7c3aed;--invalid:#667085}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e8ebf1;--mute:#98a2b3;--line:#2e3644;--accent:#7ea2ff;--cargo:#4ade80;--safety:#fb923c;--work:#c4a5ff;--invalid:#98a2b3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:16px;margin:0 8px 0 0}.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.row+.row{margin-top:8px}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;min-height:32px}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);border-color:var(--accent);color:var(--bg);font-weight:600}
.hint{color:var(--mute);font-size:13px}main{padding:14px 16px 90px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.c{background:var(--card);border:2px solid var(--line);border-radius:8px;overflow:hidden}.c.ch{border-color:var(--safety)}.c.ok{opacity:.72}
.c img{display:block;width:100%;height:160px;object-fit:contain;background:#000;cursor:zoom-in}
.m{display:flex;justify-content:space-between;gap:6px;padding:5px 8px;font-size:13px}.m b{font-variant-numeric:tabular-nums}
.b{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;padding:0 6px 6px}.b button{padding:3px 0;font-size:12.5px;min-height:30px}
.b button.cargo.on{background:var(--cargo);border-color:var(--cargo)}.b button.safety.on{background:var(--safety);border-color:var(--safety)}.b button.work.on{background:var(--work);border-color:var(--work)}.b button.invalid.on{background:var(--invalid);border-color:var(--invalid)}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}
#go{background:var(--accent);border-color:var(--accent);color:var(--bg);font-weight:700;padding:8px 18px}
#dlg{position:fixed;inset:0;z-index:9;background:rgba(0,0,0,.85);display:none;overflow:auto;padding:16px}#dlg.open{display:block}
#dlg div{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px}#dlg img{width:100%;background:#000;border-radius:6px}
#toast{position:fixed;top:12px;left:50%;transform:translateX(-50%);background:#b42318;color:#fff;padding:8px 14px;border-radius:6px;display:none;z-index:20}
.empty{padding:50px 10px;text-align:center;color:var(--mute);font-size:16px}
</style></head><body>
<header><div class="row"><h1>COUNT 라벨 검수</h1><span id="vids" class="row"></span></div><div class="row"><span id="grps" class="row"></span><label class="hint"><input type="checkbox" id="todo" checked> 미검수만 보기</label></div></header>
<main><div id="grid" class="grid"></div></main>
<footer><span id="info" class="hint"></span><span class="row"><button id="prev">이전</button><button id="next">다음</button><button id="go">이 페이지 전부 맞음 · 다음으로</button></span></footer>
<div id="dlg"><div id="big"></div></div><div id="toast" role="alert"></div>
<script>
const N={cargo:'화물',safety:'세이프티',work:'작업',invalid:'무효'},G={cargo:'화물 예측',safety:'세이프티 예측',work:'작업 예측',miss:'미탐 후보'},OP={R2L:'선적',L2R:'양하'};
let S=[],v=null,g='cargo',page=0,items=[],total=0,size=48;const $=i=>document.getElementById(i);
async function api(p,b){const r=await fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});const j=await r.json();if(!r.ok){toast(j.error||'저장 실패');throw j}return j}
function toast(t){const e=$('toast');e.textContent=t;e.style.display='block';setTimeout(()=>e.style.display='none',5000)}
function btn(t,on,f){const b=document.createElement('button');b.textContent=t;if(on)b.className='on';b.onclick=f;return b}
async function refresh(){S=await api('/api/state');v=v||S[0].video;const V=S.find(x=>x.video==v);$('vids').replaceChildren(...S.map(x=>{const d=Object.values(x.groups).reduce((a,c)=>a+c.done,0),t=Object.values(x.groups).reduce((a,c)=>a+c.total,0);return btn(`${x.video.replace('_rtspcam','')}  ${d}/${t}`,x.video==v,()=>{v=x.video;page=0;load()})}));
$('grps').replaceChildren(...Object.keys(G).filter(k=>V.groups[k].total).map(k=>btn(`${G[k]} ${V.groups[k].done}/${V.groups[k].total}`,k==g,()=>{g=k;page=0;load()})));if(!V.groups[g].total){g='cargo'}}
async function load(){await refresh();const r=await api(`/api/items?video=${encodeURIComponent(v)}&group=${g}&todo=${$('todo').checked?1:0}&page=${page}`);items=r.items;total=r.total;size=r.page_size;
if(!items.length&&page>0){page=0;return load()}draw();scrollTo(0,0)}
function draw(){const G_=$('grid');G_.className=items.length?'grid':'';if(!items.length){G_.innerHTML='<div class="empty">이 항목은 검수가 끝났습니다.</div>'}else G_.replaceChildren(...items.map(card));
const pages=Math.max(1,Math.ceil(total/size));$('info').textContent=`${total}건 남음 · ${page+1}/${pages} 페이지 · 틀린 차만 라벨을 눌러 고치세요`;$('prev').disabled=page==0;$('next').disabled=page+1>=pages;$('go').disabled=!items.length}
function card(it){const c=document.createElement('article');c.className='c'+(it.label!=it.pred?' ch':'')+(it.reviewed=='1'?' ok':'');const im=new Image();im.loading='lazy';im.alt=`${it.event} 차량`;im.src=`/img/${encodeURIComponent(v)}/${it.kind}/${it.event}/0`;im.onclick=()=>zoom(it);
const m=document.createElement('div');m.className='m';const a=document.createElement('b');a.textContent=`${it.kind=='miss'?'':'E'}${it.event}`;const t=document.createElement('span');t.textContent=`${it.time} · ${OP[it.dir]}`;m.append(a,t);
const b=document.createElement('div');b.className='b';for(const k in N){const x=document.createElement('button');x.textContent=N[k];x.className=k+(it.label==k?' on':'');x.onclick=async()=>{await api('/api/label',{video:v,kind:it.kind,id:it.event,label:k});it.label=k;it.reviewed='1';c.replaceWith(card(it));refresh()};b.append(x)}c.append(im,m,b);return c}
function zoom(it){$('big').replaceChildren(...Array.from({length:it.n},(_,i)=>{const m=new Image();m.src=`/img/${encodeURIComponent(v)}/${it.kind}/${it.event}/${i}`;m.alt='확대 이미지';return m}));$('dlg').classList.add('open')}
$('dlg').onclick=()=>$('dlg').classList.remove('open');addEventListener('keydown',e=>{if(e.key=='Escape')$('dlg').classList.remove('open')});
$('todo').onchange=()=>{page=0;load()};$('prev').onclick=()=>{page--;load()};$('next').onclick=()=>{page++;load()};
$('go').onclick=async()=>{const un=items.filter(x=>x.label==='');if(un.length){toast(`라벨을 정하지 않은 차량이 ${un.length}건 있습니다.`);return}await api('/api/confirm',{video:v,kind:items[0].kind,ids:items.map(x=>x.event)});if(!$('todo').checked&&page+1<Math.ceil(total/size))page++;load()};load();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); a = ap.parse_args()
    load(); print(f"검수 페이지: http://127.0.0.1:{a.port}  영상 {list(DB)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
