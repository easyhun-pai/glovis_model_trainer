"""2차(표식) 모델용 차량 크롭: gt_rev 의 통과 차량마다 체인 박스를 고르게 K장 샘플링해 원본 해상도로 자른다.

경광등이 잘리지 않게 위쪽 --pad-top(박스 높이 비율), 좌우 --pad-x 여유. 높이 --min-h 미만 박스 제외.
출력: data/COUNT/marks/crops/<영상>/<id>_<frame>.jpg, data/COUNT/marks/crops/<영상>.csv (file,id,label,frame,cx1,cy1,cx2,cy2,bx1..by2)
홀드아웃('검증용')은 거부한다.   사용: extract_mark_crops.py <영상>... [--k 8]
"""
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--min-h", type=int, default=150); ap.add_argument("--pad-top", type=float, default=0.3); ap.add_argument("--pad-x", type=float, default=0.06)
a = ap.parse_args()
for v in a.videos:
    assert "검증용" not in v, "홀드아웃 영상은 사용 금지"
    gt = pd.read_csv(ROOT / "data/COUNT/gt_rev" / f"{v}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gt = gt[(gt.note == "") & (gt.label != "invalid")]
    tr = pd.read_csv(ROOT / "runs/count" / v / "tracks_split.csv"); tr = tr[(tr.y2 - tr.y1) >= a.min_h]
    want = {}
    for r in gt.itertuples():
        g = tr[tr.tid.isin([int(t) for t in r.chain.split("|")])].sort_values("frame")
        if g.empty: continue
        sel = g.iloc[np.unique(np.linspace(0, len(g) - 1, min(a.k, len(g))).round().astype(int))]
        for b in sel.itertuples(): want.setdefault(int(b.frame), []).append((r.id, r.label or "cargo?", b))
    out = ROOT / "data/COUNT/marks/crops" / v; out.mkdir(parents=True, exist_ok=True); rows = []
    cap = cv2.VideoCapture(str(ROOT / "data/COUNT" / f"{v}.mp4")); fi = -1; last = max(want)
    while fi < last:
        if not cap.grab(): break
        fi += 1
        if fi not in want: continue
        im = cap.retrieve()[1]; H, W = im.shape[:2]
        for i, lab, b in want[fi]:
            w, h = b.x2 - b.x1, b.y2 - b.y1
            x1, y1 = max(0, int(b.x1 - a.pad_x * w)), max(0, int(b.y1 - a.pad_top * h)); x2, y2 = min(W, int(b.x2 + a.pad_x * w)), min(H, int(b.y2))
            f = f"{i}_{fi}.jpg"; cv2.imwrite(str(out / f), im[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(dict(file=f, id=i, label=lab, frame=fi, cx1=x1, cy1=y1, cx2=x2, cy2=y2, bx1=b.x1, by1=b.y1, bx2=b.x2, by2=b.y2))
    pd.DataFrame(rows).to_csv(out.parent / f"{v}.csv", index=False)
    print(v, len(gt), "vehicles", len(rows), "crops", flush=True)
