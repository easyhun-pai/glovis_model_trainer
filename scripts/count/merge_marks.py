"""표식 라벨 2판(draft_r2): draft_r1(keep2) + draft_vp2(멀티 참조 시각 프롬프트)를 합치고 규칙으로 거른다.

- vp2 기준: 경광등 conf >= --beacon-th, 번호 conf >= --number-th, 박스 면적 < 크롭 5%
- 소유 위치 규칙(filter_marks 와 동일): 경광등은 차량 박스 윗변 근처, 번호는 박스 위 60%, 번호판은 박스 안
- 표식 이동 규칙(사용자 결정 2026-11-24, 경광등·번호·번호판 모두): 같은 차량의 다른 크롭에서 전체 프레임 좌표가 거의 같은 같은 종류 표식이 있는데
  그 사이 차량은 크게 움직였다면 고정 구조물(과속 경고등·볼라드·표지판 숫자 등) → 버림
- 중복은 같은 클래스 IoU >= 0.4 면 conf 높은 것만
출력: data/COUNT/marks/draft_r2/<영상>.csv (make_marks_dataset --labels-dir 용, keep2=1)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"
ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--beacon-th", type=float, default=0.3); ap.add_argument("--number-th", type=float, default=0.2)
ap.add_argument("--out", default="draft_r2"); a = ap.parse_args(); (MK / a.out).mkdir(exist_ok=True)
COLS = ["file", "id", "label", "cls", "conf", "x1", "y1", "x2", "y2", "W", "H", "src"]


def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-9)


for v in a.videos:
    c = pd.read_csv(MK / "crops" / f"{v}.csv").set_index("file")
    r1 = pd.read_csv(MK / "draft_r1" / f"{v}.csv"); r1 = r1[r1.keep2 == 1].assign(src="r1")
    vp = pd.read_csv(MK / "draft_vp2" / f"{v}.csv").assign(src="vp2")
    vp = vp[((vp.cls == 1) & (vp.conf >= a.beacon_th)) | ((vp.cls == 2) & (vp.conf >= a.number_th))]
    vp = vp[(vp.x2 - vp.x1) * (vp.y2 - vp.y1) < 0.05 * vp.W * vp.H]
    d = pd.concat([r1[COLS], vp[COLS]], ignore_index=True)
    d = d.join(c.assign(X1=c.bx1 - c.cx1, Y1=c.by1 - c.cy1, X2=c.bx2 - c.cx1, Y2=c.by2 - c.cy1)[["X1", "Y1", "X2", "Y2", "cx1", "cy1", "bx1", "bx2", "by1", "by2"]], on="file")
    mx, my, h = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, d.Y2 - d.Y1; inx = (mx >= d.X1) & (mx <= d.X2)
    own = ((d.cls == 1) & inx & (my >= d.Y1 - 0.25 * h) & (my <= d.Y1 + 0.20 * h)) | ((d.cls == 2) & inx & (my >= d.Y1) & (my <= d.Y1 + 0.6 * h)) | \
          ((d.cls == 0) & inx & (my >= d.Y1) & (my <= d.Y2))
    d = d[own].copy()
    # 경광등 이동 규칙
    d["gx"], d["gy"] = d.cx1 + (d.x1 + d.x2) / 2, d.cy1 + (d.y1 + d.y2) / 2; d["vx"] = (d.bx1 + d.bx2) / 2; d["vw"] = d.bx2 - d.bx1
    d["vh"] = d.by2 - d.by1; d["rx"], d["ry"] = (d.gx - d.bx1) / d.vw, (d.gy - d.by1) / d.vh
    static = set()
    for _, g in d.groupby(["id", "cls"]):  # 세 표식 모두 (사용자 결정 2026-11-24)
        P = g[["gx", "gy", "bx1", "bx2", "vw", "rx", "ry"]].to_numpy(); idx = g.index.to_numpy(); fs = g.file.to_numpy()
        for i in range(len(P)):
            fixed = follow = False
            for j in range(len(P)):
                if fs[i] == fs[j]: continue
                # 차량이 실제로 움직였는지: 박스 양 끝(x1,x2)이 같은 방향으로 0.25w 이상 이동 (가림으로 박스만 줄어든 경우 제외)
                d1, d2 = P[j, 2] - P[i, 2], P[j, 3] - P[i, 3]; moved = d1 * d2 > 0 and min(abs(d1), abs(d2)) > 0.25 * P[i, 4]
                if not moved: continue
                if np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]) < 0.04 * P[i, 4]: fixed = True      # 화면 좌표 고정
                if abs(P[i, 5] - P[j, 5]) < 0.10 and abs(P[i, 6] - P[j, 6]) < 0.10: follow = True     # 차량 기준 같은 자리 = 차와 같이 이동
            # 같은 종류 표식이 차에 여러 개(경광등 2, 번호 3)라 우연히 화면 좌표가 겹칠 수 있음 → 차와 같이 움직인 짝이 없을 때만 고정물로 판단
            if fixed and not follow: static.add(idx[i])
    d.loc[list(static)].to_csv(MK / a.out / f"{v}_static.csv", index=False); d = d.drop(index=list(static))
    # 중복 제거
    keep = []
    for f, g in d.sort_values("conf", ascending=False).groupby("file", sort=False):
        kept = []
        for r in g.itertuples():
            b = (r.x1, r.y1, r.x2, r.y2)
            if any(k.cls == r.cls and iou(b, (k.x1, k.y1, k.x2, k.y2)) >= 0.4 for k in kept): continue
            kept.append(r); keep.append(r.Index)
    d = d.loc[keep, COLS].assign(keep2=1); d.to_csv(MK / a.out / f"{v}.csv", index=False)
    s = d[d.label.isin(["safety", "work"])]; n = c[c.label.isin(["safety", "work"])].shape[0]
    print(v, f"static marks dropped {len(static)}", "| non-cargo crops", n, "| crops with cls:", s.groupby("cls").file.nunique().to_dict(), "| boxes by src:", s.groupby(["cls", "src"]).size().to_dict())
