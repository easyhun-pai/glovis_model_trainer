"""3차 표식 라벨: 사용자 정답(gt_edit.csv) + 나머지 크롭은 자동 라벨(r2 모델 예측 ∪ draft_r2m)을 규칙으로 거른 것.

규칙: 소유 위치(merge_marks 와 동일) → 표식 이동 규칙(세 표식) → 클래스별 conf 기준 → 번호 일관성
  (번호는 conf >= hi 이거나, conf >= lo 이면서 같은 차량 다른 크롭의 차량 기준 같은 자리에도 번호가 있을 때만)
기준값은 학습 영상의 정답 크롭(--tune-videos)에서 F1 이 최대가 되게 고른다(평가 영상 260818_s1 정답은 쓰지 않음).
출력: data/COUNT/marks/labels_r3/<영상>.csv (keep2=1; 정답 크롭은 정답 박스로 대체, src=gt)
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from eval_marks import score
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"
ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--tune-videos", nargs="+", default=["260731_rtspcam", "260818_s2_rtspcam"])
ap.add_argument("--out", default="labels_r3"); a = ap.parse_args()
COLS = ["file", "id", "label", "cls", "conf", "x1", "y1", "x2", "y2", "W", "H", "src"]


def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-9)


def candidates(v):
    c = pd.read_csv(MK / "crops" / f"{v}.csv").set_index("file")
    p = pd.read_csv(MK / "pred_r2" / f"{v}.csv"); d = pd.read_csv(MK / "draft_r2m" / f"{v}.csv")
    d = d.assign(conf=d.conf.clip(upper=0.99))  # 초안(r1 채움·시각프롬프트) 점수
    x = pd.concat([p[COLS], d[COLS]], ignore_index=True)
    # 같은 크롭·클래스 IoU>=0.5 는 하나로 (점수 큰 것), 두 출처가 겹치면 점수 +0.15
    keep, bonus = [], {}
    for _, g in x.sort_values("conf", ascending=False).groupby(["file", "cls"], sort=False):
        kept = []
        for r in g.itertuples():
            hit = [k for k in kept if iou((r.x1, r.y1, r.x2, r.y2), (k.x1, k.y1, k.x2, k.y2)) >= 0.5]
            if hit:
                if hit[0].src != r.src: bonus[hit[0].Index] = 0.15
                continue
            kept.append(r); keep.append(r.Index)
    x = x.loc[keep].copy(); x["conf"] = (x.conf + pd.Series(bonus).reindex(x.index).fillna(0)).clip(upper=1)
    x = x.join(c.assign(X1=c.bx1 - c.cx1, Y1=c.by1 - c.cy1, X2=c.bx2 - c.cx1, Y2=c.by2 - c.cy1)[["X1", "Y1", "X2", "Y2", "cx1", "cy1", "bx1", "bx2", "by1", "by2"]], on="file")
    mx, my, h = (x.x1 + x.x2) / 2, (x.y1 + x.y2) / 2, x.Y2 - x.Y1; inx = (mx >= x.X1) & (mx <= x.X2)
    own = ((x.cls == 1) & inx & (my >= x.Y1 - 0.25 * h) & (my <= x.Y1 + 0.20 * h)) | ((x.cls == 2) & inx & (my >= x.Y1) & (my <= x.Y1 + 0.6 * h)) | ((x.cls == 0) & inx & (my >= x.Y1) & (my <= x.Y2))
    x = x[own & ((x.x2 - x.x1) * (x.y2 - x.y1) < 0.05 * x.W * x.H)].copy()
    x["gx"], x["gy"] = x.cx1 + (x.x1 + x.x2) / 2, x.cy1 + (x.y1 + x.y2) / 2; x["vw"], x["vh"] = x.bx2 - x.bx1, x.by2 - x.by1
    x["rx"], x["ry"] = (x.gx - x.bx1) / x.vw, (x.gy - x.by1) / x.vh
    st, fl = {}, {}
    for _, g in x.groupby(["id", "cls"]):
        P = g[["gx", "gy", "bx1", "bx2", "vw", "rx", "ry"]].to_numpy(); idx = g.index.to_numpy(); fs = g.file.to_numpy()
        for i in range(len(P)):
            fixed = follow = False
            for j in range(len(P)):
                if fs[i] == fs[j]: continue
                if abs(P[i, 5] - P[j, 5]) < 0.10 and abs(P[i, 6] - P[j, 6]) < 0.10: follow = True
                d1, d2 = P[j, 2] - P[i, 2], P[j, 3] - P[i, 3]
                if d1 * d2 > 0 and min(abs(d1), abs(d2)) > 0.25 * P[i, 4] and np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]) < 0.04 * P[i, 4]: fixed = True
            st[idx[i]] = fixed and not follow; fl[idx[i]] = follow
    x["static"] = pd.Series(st); x["follow"] = pd.Series(fl)
    return x[~x.static]


def select(x, th):
    t0, t1, lo, hi = th
    return x[((x.cls == 0) & (x.conf >= t0)) | ((x.cls == 1) & (x.conf >= t1)) | ((x.cls == 2) & ((x.conf >= hi) | ((x.conf >= lo) & x.follow)))]


C = {v: candidates(v) for v in a.videos}
g = pd.read_csv(MK / "gt_edit.csv"); gt = {}
for (v, f), q in g.groupby(["video", "file"]): gt[f"{v}/{f}"] = [(int(r.cls), r.x1, r.y1, r.x2, r.y2) for r in q.itertuples() if r.cls >= 0]
tune = {k: b for k, b in gt.items() if k.split("/")[0] in a.tune_videos}
best = {}
for th in itertools.product([0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]):
    if th[2] > th[3]: continue
    pred = {}
    for v, x in C.items():
        for f, q in select(x, th).groupby("file"):
            if f"{v}/{f}" in tune: pred[f"{v}/{f}"] = [(int(r.cls), r.x1, r.y1, r.x2, r.y2) for r in q.itertuples()]
    S = score(tune, pred, 0.5)
    for k, key in ((0, (0,)), (1, (1,)), (2, (2, 3))):
        tp, fp, fn = S[k]; f1 = 2 * tp / max(1, 2 * tp + fp + fn); sel = tuple(th[i] for i in key)
        if f1 > best.get(k, (0,))[0]: best[k] = (f1, sel, tp / max(1, tp + fp), tp / max(1, tp + fn))
for k, n in enumerate(["plate", "beacon", "number"]): print(f"{n}: F1 {best[k][0]:.3f} th {best[k][1]} P {best[k][2]:.2f} R {best[k][3]:.2f}  (tune crops {len(tune)})")
th = (best[0][1][0], best[1][1][0], best[2][1][0], best[2][1][1])
(MK / a.out).mkdir(exist_ok=True)
for v, x in C.items():
    y = select(x, th)[COLS].copy(); gk = {k.split("/")[1] for k in gt if k.startswith(v + "/")}
    y = y[~y.file.isin(gk)]
    q = g[(g.video == v) & (g.cls >= 0)]; c = pd.read_csv(MK / "crops" / f"{v}.csv").set_index("file")
    G = pd.DataFrame(dict(file=q.file, id=c.loc[q.file].id.values, label=c.loc[q.file].label.values, cls=q.cls, conf=1.0, x1=q.x1, y1=q.y1, x2=q.x2, y2=q.y2, src="gt"))
    if len(G):
        import cv2
        wh = {f: cv2.imread(str(MK / "crops" / v / f)).shape[:2] for f in G.file.unique()}; G["W"] = G.file.map(lambda f: wh[f][1]); G["H"] = G.file.map(lambda f: wh[f][0])
    y = pd.concat([y, G[COLS]] if len(G) else [y], ignore_index=True).assign(keep2=1); y.to_csv(MK / a.out / f"{v}.csv", index=False)
    s = y[y.label.isin(["safety", "work"])]; print(v, "boxes by cls/src:", s.groupby(["cls", "src"]).size().to_dict(), "| gt crops", len(gk))
print("thresholds (plate, beacon, number lo, number hi):", th)
