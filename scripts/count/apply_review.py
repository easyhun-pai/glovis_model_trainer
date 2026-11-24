"""웹 검수 결과(Artifact db reviews/<영상> 의 m 맵, JSON 으로 받아둔 것) → data/COUNT/gt_rev/<영상>.csv

열: id, time(mm:ss), time_s, dir, op, label(cargo/safety/work/invalid), chain(tid|tid), frame, note
- E 번호(숫자 id)는 runs/count/<영상>/events_v1.csv 기준, T<tid> 는 미탐 후보, N<frame> 은 events.csv 기준 추가 발견.
- DUP: 같은 차가 두 이벤트로 잡힌 것 → label 은 두되 note=dup 로 집계 제외. MERGE: 미탐 후보 두 조각이 같은 차.
사용: apply_review.py <db json 폴더>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data/COUNT/gt_rev"; OUT.mkdir(exist_ok=True)
DUP = {("260818_s2_rtspcam", "468"): "dup of E466 (ID switch, same car)"}
MERGE = {("260818_s2_rtspcam", "T15945"): "T15932"}  # 사용자 확인 2026-11-23: 선 위에서 가려져 두 조각
items = json.load(open(ROOT / "runs/count/review/web/items.json", encoding="utf8"))["items"]
db = Path(sys.argv[1])
for v in sorted({x["v"] for x in items}):
    m = json.load(open(db / f"{v}.json", encoding="utf8"))["m"]; run = ROOT / "runs/count" / v
    old = pd.read_csv(run / "events_v1.csv", dtype={"event": str}).set_index("event")
    new = pd.read_csv(run / "events.csv").set_index("frame"); pts = np.load(run / "pts.npy")
    tr = pd.read_csv(run / "tracks_split.csv"); rows = []
    for x in items:
        if x["v"] != v: continue
        i = x["id"]; lab = m.get(i, ""); note = DUP.get((v, i), "")
        if i.startswith("T"):
            if (v, i) in MERGE: continue
            ch = [int(i[1:])] + [int(k[1][1:]) for k, t in MERGE.items() if k[0] == v and t == i]
            g = tr[tr.tid.isin(ch)]; fr = int(g.frame.median()); src = "miss"
        elif i.startswith("N"):
            fr = int(i[1:]); ch = [int(t) for t in str(new.chain[fr]).split("|")]; src = "new"
        else:
            fr = int(old.frame[i]); ch = [int(t) for t in str(old.chain[i]).split("|")]; src = "event"
        ts = float(old.time_s[i]) if src == "event" else float(pts[min(fr, len(pts) - 1)])
        rows.append(dict(id=i, time=x["t"], time_s=round(ts, 1), dir=x["d"], op={"R2L": "load", "L2R": "unload"}[x["d"]],
                         label=lab, chain="|".join(map(str, ch)), frame=fr, src=src, note=note))
    d = pd.DataFrame(rows).sort_values("time_s", kind="stable"); d.to_csv(OUT / f"{v}.csv", index=False, encoding="utf-8-sig")
    ok = d[(d.note == "") & (d.label != "invalid")]
    print(v, len(d), "rows | counted", len(ok), ok.groupby(["op", "label"]).size().to_dict(), "| unlabeled", int((d.label == "").sum()))
