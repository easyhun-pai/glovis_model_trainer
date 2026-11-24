"""미탐 진단: 우측 진입(첫 위치 x > --entry-x, 좌향 이동) 트랙 중 어떤 이벤트 체인에도 속하지 않은 것 → 미탐 후보.

탐지/점유와 독립적인 교차검증. 구름 그림자로 배경차분 점유 검사가 무력한 영상에서 사용.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--entry-x", type=float, default=2100)
ap.add_argument("--min-h", type=float, default=150)
a = ap.parse_args(); run = Path(a.run); fps = json.load(open(run / "meta.json"))["fps"]
t = pd.read_csv(run / "tracks.csv"); t = t[(t.y2 - t.y1) >= a.min_h]
ev = pd.read_csv(run / "events.csv")
in_ev = set(int(x) for c in ev.chain.astype(str) for x in c.split("|"))
cand = []
for tid, g in t.groupby("tid"):
    g = g.sort_values("frame"); bx = (g.x1 + g.x2) / 2
    if len(g) >= 2 and bx.iloc[0] > a.entry_x and bx.iloc[-1] < bx.iloc[0] - 100 and tid not in in_ev:
        cand.append((tid, g.frame.iloc[0], round(g.frame.iloc[0] / fps, 1), int(bx.iloc[0]), int(bx.iloc[-1]), len(g)))
print(f"{run.name}: 이벤트에 안 붙은 우측 진입 트랙 {len(cand)}개")
for c in cand: print("  tid=%d f=%d t=%.1fs x %d->%d n=%d" % c)
