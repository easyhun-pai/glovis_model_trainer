"""자기학습 채움: 표식 모델로 크롭 전체를 예측해 초안에 없는 박스(IoU<0.3)를 추가 → data/COUNT/marks/<out>/<영상>.csv
초안 박스는 그대로 두고 src=draft / model 로 구분. 이후 filter_marks 와 같은 위치 규칙을 적용(keep2).
또 차량 단위 판정(표식 conf>=--veh-conf 인 크롭 수 >= --min-frames → 제외)을 gt_rev 와 비교해 출력.
사용: fill_marks.py --model best.pt --out draft_r1 영상...
"""
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"
ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--model", required=True); ap.add_argument("--out", default="draft_r1")
ap.add_argument("--conf", type=float, default=0.4); ap.add_argument("--src", default="draft"); ap.add_argument("--veh-conf", type=float, default=0.5); ap.add_argument("--min-frames", type=int, default=2)
a = ap.parse_args(); m = YOLO(a.model); (MK / a.out).mkdir(exist_ok=True)


def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-9)


for v in a.videos:
    c = pd.read_csv(MK / "crops" / f"{v}.csv"); d = pd.read_csv(MK / a.src / f"{v}.csv"); d = d[d.keep2 == 1].assign(src="draft")
    g = {f: x for f, x in d.groupby("file")}; add = []; pred = []
    for s in range(0, len(c), 32):
        b = c.iloc[s:s + 32]; R = m.predict([str(MK / "crops" / v / f) for f in b.file], imgsz=640, conf=min(a.conf, a.veh_conf, 0.25), verbose=False)
        for r, res in zip(b.itertuples(), R):
            H, W = res.orig_shape
            for xy, cf, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                pred.append((r.file, r.id, int(k), cf))
                if cf < a.conf: continue
                old = g.get(r.file)
                if old is not None and any(iou(xy, (q.x1, q.y1, q.x2, q.y2)) >= 0.3 for q in old.itertuples()): continue
                add.append(dict(file=r.file, id=r.id, label=r.label, cls=int(k), conf=round(cf, 3), x1=round(xy[0]), y1=round(xy[1]), x2=round(xy[2]), y2=round(xy[3]), W=W, H=H, keep=1, src="model"))
    A = pd.DataFrame(add)
    # 위치 규칙 (filter_marks 와 동일)
    cc = c.set_index("file"); A = A.join(cc.assign(X1=cc.bx1 - cc.cx1, Y1=cc.by1 - cc.cy1, X2=cc.bx2 - cc.cx1, Y2=cc.by2 - cc.cy1)[["X1", "Y1", "X2", "Y2"]], on="file")
    mx, my, h = (A.x1 + A.x2) / 2, (A.y1 + A.y2) / 2, A.Y2 - A.Y1; inx = (mx >= A.X1) & (mx <= A.X2)
    A["keep2"] = (((A.cls == 1) & inx & (my >= A.Y1 - 0.25 * h) & (my <= A.Y1 + 0.20 * h)) | ((A.cls == 2) & inx & (my >= A.Y1) & (my <= A.Y1 + 0.6 * h)) |
                  ((A.cls == 0) & inx & (my >= A.Y1) & (my <= A.Y2))).astype(int)
    out = pd.concat([d, A]); out.to_csv(MK / a.out / f"{v}.csv", index=False)
    k = A[A.keep2 == 1]; print(v, "added (label,cls):", k.groupby(["label", "cls"]).size().to_dict())
    # 차량 단위 판정
    P = pd.DataFrame(pred, columns=["file", "id", "cls", "conf"]); P = P[P.conf >= a.veh_conf]
    nf = P.groupby("id").file.nunique(); lab = c.drop_duplicates("id").set_index("id").label
    ex = lab.index.map(lambda i: nf.get(i, 0) >= a.min_frames); gt_ex = lab.isin(["safety", "work"])
    print(f"  vehicle: n={len(lab)} cargo->excluded {int((ex & ~gt_ex).sum())}/{int((~gt_ex).sum())}  noncargo->kept {int((~ex & gt_ex).sum())}/{int(gt_ex.sum())}")
    print("   FP ids", list(lab.index[ex & ~gt_ex])[:20], " FN ids", list(lab.index[~ex & gt_ex])[:20])
