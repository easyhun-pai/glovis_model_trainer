"""클라우드 검수 페이지용 정적 데이터 생성: gt_draft + 크롭 → runs/count/review/web/{items.json, a/*.jpg}.

차량당 2컷(주 이미지, 다른 시점)을 아틀라스(48대씩)로 묶는다. 홀드아웃('검증용')은 처리하지 않는다.
"""
import glob, json, os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "data/COUNT/gt_draft"; OUT = ROOT / "runs/count/review/web"; (OUT / "a").mkdir(parents=True, exist_ok=True)
TW, TH, COLS, ROWS = 480, 340, 8, 6
for f in (OUT / "a").glob("*.jpg"): f.unlink()


def fit(im):
    t = np.full((TH, TW, 3), 16, np.uint8)
    if im is None: return t
    s = min(TW / im.shape[1], TH / im.shape[0]); q = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    y, x = (TH - q.shape[0]) // 2, (TW - q.shape[1]) // 2; t[y:y + q.shape[0], x:x + q.shape[1]] = q
    return t


def views(paths):
    if not paths: return None, None
    paths = sorted(paths); ims = [cv2.imread(p) for p in paths]
    i = int(np.argmax([m.shape[0] * min(m.shape[1], m.shape[0] * 1.7) for m in ims]))
    j = int(np.argmax([abs(k - i) for k in range(len(ims))]))
    return ims[i], ims[j] if j != i else None


items, tiles = [], []
def flush():
    n = len(tiles) // (COLS * ROWS) if False else 0


atlas = 0; buf = []
def push(a, b):
    global atlas, buf
    buf += [fit(a), fit(b)]
    if len(buf) == COLS * ROWS: write()


def write():
    global atlas, buf
    if not buf: return
    while len(buf) < COLS * ROWS: buf.append(np.full((TH, TW, 3), 16, np.uint8))
    cv2.imwrite(str(OUT / "a" / f"{atlas:03d}.jpg"), np.vstack([np.hstack(buf[i:i + COLS]) for i in range(0, len(buf), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 76])
    atlas += 1; buf = []


for p in sorted(DRAFT.glob("*.csv")):
    v = p.stem
    if v.endswith("_miss") or "검증용" in v: continue
    d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig"); run = ROOT / "runs/count" / v
    chains = pd.read_csv(run / "events.csv", dtype=str).set_index("event").chain.to_dict()
    for g in ("cargo", "safety", "work"):
        for r in d[d.label == g].itertuples():
            c = []
            for t in chains[r.event].split("|"): c += glob.glob(str(run / "crops" / str(int(t) % 1000000) / "*.jpg"))
            k = len(buf) // 2; items.append(dict(v=v, id=r.event, g=g, t=r.time, d=r.dir, p=g, a=atlas, i=k)); push(*views(c))
    m = DRAFT / f"{v}_miss.csv"
    if m.exists():
        for r in pd.read_csv(m, dtype=str, keep_default_na=False, encoding="utf-8-sig").itertuples():
            b = ROOT / "runs/count/review" / v / "miss_ctx" / r.event
            k = len(buf) // 2; items.append(dict(v=v, id=r.event, g="miss", t=r.time, d=r.dir, p=r.label, a=atlas, i=k))
            push(cv2.imread(f"{b}_ctx.jpg"), cv2.imread(f"{b}_crop.jpg"))
write()
json.dump(dict(items=items, tile=[TW, TH], grid=[COLS, ROWS], atlases=atlas), open(OUT / "items.json", "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
s = sum(f.stat().st_size for f in (OUT / "a").glob("*.jpg"))
print(len(items), "items", atlas, "atlases", round(s / 1048576, 1), "MB", "max", round(max(f.stat().st_size for f in (OUT / "a").glob("*.jpg")) / 1048576, 2), "MB")
