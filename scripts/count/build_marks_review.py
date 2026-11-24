"""표식 초안 샘플 검수 페이지 데이터: runs/count/review/marks/{items.json, a/NNN.jpg}

sv: 세이프티·작업 크롭에서 차량당 최대 1장, --n-sv 장 (박스 유무 섞이게 무작위)
cg: 화물 크롭 중 keep2 박스가 있는 것에서 --n-cg 장
타일 480x400, 박스는 원본 좌표로 그린 뒤 축소. 아틀라스 4x4.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "runs/count/review/marks"; (OUT / "a").mkdir(parents=True, exist_ok=True)
TW, TH, C, R = 480, 400, 4, 4; COL = {0: (44, 53, 232), 1: (0, 176, 255), 2: (251, 64, 224)}
ap = argparse.ArgumentParser(); ap.add_argument("--n-sv", type=int, default=240); ap.add_argument("--n-cg", type=int, default=72); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--src", default="draft_r1")
a = ap.parse_args(); V = ["260731_rtspcam", "260818_s1_rtspcam", "260818_s2_rtspcam"]
crops = pd.concat([pd.read_csv(ROOT / "data/COUNT/marks/crops" / f"{v}.csv").assign(v=v) for v in V])
det = pd.concat([pd.read_csv(ROOT / "data/COUNT/marks" / a.src / f"{v}.csv").assign(v=v) for v in V]); det = det[det.keep2 == 1]
sv = crops[crops.label.isin(["safety", "work"])].sample(frac=1, random_state=a.seed).drop_duplicates(["v", "id"]).head(a.n_sv).assign(t="sv")
cgf = det[det.label.isin(["cargo", "cargo?"])][["v", "file"]].drop_duplicates()
cg = crops.merge(cgf, on=["v", "file"]).sample(frac=1, random_state=a.seed).drop_duplicates(["v", "id"]).head(a.n_cg).assign(t="cg")
S = pd.concat([sv, cg]).reset_index(drop=True)
for f in (OUT / "a").glob("*.jpg"): f.unlink()
items, buf, atlas = [], [], 0
def write():
    global buf, atlas
    while len(buf) < C * R: buf.append(np.full((TH, TW, 3), 16, np.uint8))
    cv2.imwrite(str(OUT / "a" / f"{atlas:03d}.jpg"), np.vstack([np.hstack(buf[i:i + C]) for i in range(0, C * R, C)]), [cv2.IMWRITE_JPEG_QUALITY, 82]); atlas += 1; buf = []
for r in S.itertuples():
    im = cv2.imread(str(ROOT / "data/COUNT/marks/crops" / r.v / r.file)); lw = max(2, int(round(im.shape[0] / 200)))
    for q in det[(det.v == r.v) & (det.file == r.file)].itertuples(): cv2.rectangle(im, (q.x1 - lw, q.y1 - lw), (q.x2 + lw, q.y2 + lw), COL[q.cls], lw)
    s = min(TW / im.shape[1], TH / im.shape[0]); q = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = np.full((TH, TW, 3), 16, np.uint8); y, x = (TH - q.shape[0]) // 2, (TW - q.shape[1]) // 2; t[y:y + q.shape[0], x:x + q.shape[1]] = q
    items.append(dict(k=f"{r.v}/{r.file}", t=r.t, n=f"{r.v[:9].replace('_rtspcam','')} · {r.id}", s={"safety": "세이프티", "work": "작업"}.get(r.label, "화물"), a=atlas, i=len(buf)))
    buf.append(t)
    if len(buf) == C * R: write()
if buf: write()
json.dump(dict(items=items, grid=[C, R]), open(OUT / "items.json", "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
print(len(items), "items", S.t.value_counts().to_dict(), atlas, "atlases", round(sum(f.stat().st_size for f in (OUT / "a").glob("*.jpg")) / 1048576, 1), "MB")
