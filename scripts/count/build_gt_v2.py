"""사용자 검수본(data/COUNT/gt/*.csv, 원본 유지) → 평가용 data/COUNT/gt_v2/*.csv.

- 사용자 메모(// 주석, '놓친거' 같은 글 줄)를 걷어내고 추가된 행은 반영
- time_s 를 플레이어와 같은 실제 재생 시각 + 새 기준선 통과 시점으로 변환 (같은 방향 ±6초 1:1 최적 대응)
- reviewed: 사용자가 검수한 행 1, 아직 안 본 행 0 (s4 는 메모가 달린 230.9초 행까지만 검수됨)
- time: 플레이어와 같은 mm:ss, safety_no: 정수
- op: load(선적, 야드→배) / unload(하역, 배→야드)
"""
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
MAX_DT = 6.0  # 초
ROW = re.compile(r"^\s*([\d.]+)\s*,\s*(R2L|L2R)\s*,\s*(\w+)\s*,\s*([\d.]*)\s*$")

for src in sorted((ROOT / "data/COUNT/gt").glob("*.csv")):
    v = src.stem; run = ROOT / "runs/count" / v
    fps = json.load(open(run / "meta.json"))["fps"]; pts = np.load(run / "pts.npy"); ev = pd.read_csv(run / "events.csv")
    rows, added, reviewed = [], False, True
    for line in src.read_text(encoding="utf8").splitlines()[1:]:
        body, memo = (line.split("//", 1) + [None])[:2]
        m = ROW.match(body)
        if not m:
            if body.strip(): added = True  # '놓친거' 이후 행은 사용자가 플레이어 시각으로 직접 적은 것
            continue
        t, d, label, no = float(m[1]), m[2], m[3], m[4]
        t = t if added else float(pts[min(int(round(t * fps)), len(pts) - 1)])
        rows.append(dict(t=t, dir=d, label=label, safety_no=int(float(no)) if no else "", reviewed=int(reviewed)))
        if memo is not None: reviewed = False
    rows.sort(key=lambda x: x["t"]); match, used = {}, set()
    for d in ("R2L", "L2R"):  # 방향별 전체 시간차 합이 최소가 되도록 1:1 대응 (앞에서부터 고르면 한 칸씩 밀림)
        ri = [i for i, r in enumerate(rows) if r["dir"] == d]; e = ev[ev.dir == d]
        if not ri or e.empty: continue
        C = np.abs(np.subtract.outer([rows[i]["t"] for i in ri], e.time_s.values)); C2 = np.where(C <= MAX_DT, C, 1e4)
        for i, j in zip(*linear_sum_assignment(C2)):
            if C[i, j] <= MAX_DT: match[ri[i]] = float(e.time_s.iloc[j]); used.add(int(e.event.iloc[j]))
    out, lost = [], []
    for i, r in enumerate(rows):
        if i not in match: lost.append(round(r["t"], 1))
        out.append(dict(time_s=match.get(i, round(r["t"], 1)), dir=r["dir"], op="load" if r["dir"] == "R2L" else "unload",
                        label=r["label"], safety_no=r["safety_no"], reviewed=r["reviewed"]))
    (ROOT / "data/COUNT/gt_v2").mkdir(exist_ok=True)
    df = pd.DataFrame(out).sort_values("time_s", kind="stable")
    df.insert(0, "time", [f"{int(round(t)) // 60:02d}:{int(round(t)) % 60:02d}" for t in df.time_s])
    df["safety_no"] = df.safety_no.map(lambda x: "" if x == "" else str(int(x)))
    df.drop(columns="time_s").to_csv(ROOT / "data/COUNT/gt_v2" / f"{v}.csv", index=False)
    print(f"{v}: {len(out)}행, 검수 {sum(x['reviewed'] for x in out)}행, 이벤트 미대응 {lost}, GT 없는 이벤트 {ev[~ev.event.isin(used)].time_s.tolist()}")
