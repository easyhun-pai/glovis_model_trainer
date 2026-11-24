"""검수 초안 생성: audit_labels.csv → data/COUNT/gt_draft/<영상>.csv + runs/count/review/<영상>/ 이미지 시트.

시트는 예측 라벨별로 묶여 있어, 사용자는 잘못 묶인 차량의 E번호만 찾으면 된다.
miss_*.jpg 는 기준선 통과로 세지 않았지만 크게 이동한 트랙(미탐 후보)이다.
검증용(홀드아웃) 영상은 처리하지 않는다.
"""
import glob, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W, H, COLS, ROWS = 235, 170, 8, 6


def mmss(t): t = int(round(t)); return f"{t // 60:02d}:{t % 60:02d}"


def crop_of(run, chain):
    c = []
    for t in str(chain).split("|"): c += glob.glob(str(run / "crops" / str(int(t) % 1000000) / "*.jpg"))
    if not c: return None
    return max((cv2.imread(x) for x in sorted(c)), key=lambda i: i.shape[0] * min(i.shape[1], i.shape[0] * 1.7))


def tile(im, text):
    t = np.full((H, W, 3), 30, np.uint8)
    if im is not None:
        s = min(W / im.shape[1], (H - 22) / im.shape[0]); q = cv2.resize(im, None, fx=s, fy=s)
        t[22:22 + q.shape[0], :q.shape[1]] = q
    cv2.putText(t, text, (3, 16), 0, .5, (0, 255, 255), 1)
    return t


def save(tiles, out, name):
    n = COLS * ROWS
    for k in range(0, len(tiles), n):
        T = tiles[k:k + n]
        while len(T) % COLS: T.append(np.full((H, W, 3), 30, np.uint8))
        cv2.imwrite(str(out / f"{name}_{k // n:02d}.jpg"), np.vstack([np.hstack(T[i:i + COLS]) for i in range(0, len(T), COLS)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def frame_crop(video, t, b):
    p = ROOT / "runs/count/review/_f.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(p)], check=True)
    im = cv2.imread(str(p)); return im[max(0, int(b.y1)):int(b.y2), max(0, int(b.x1)):int(b.x2)]


for v in sys.argv[1:]:
    if "검증용" in v: raise SystemExit("홀드아웃 영상은 처리 금지")
    run = ROOT / "runs/count" / v; out = ROOT / "runs/count/review" / v; out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(run / "audit_labels.csv").merge(pd.read_csv(run / "events.csv")[["event", "chain"]], on="event").sort_values("time_s")
    d = pd.DataFrame(dict(event=a.event, time=a.time_s.map(mmss), dir=a.dir, op=a.dir.map({"R2L": "load", "L2R": "unload"}), label=a.audit,
                          safety_no="", plate_conf=a.plate_conf.round(2), reviewed=0))
    (ROOT / "data/COUNT/gt_draft").mkdir(exist_ok=True); d.to_csv(ROOT / "data/COUNT/gt_draft" / f"{v}.csv", index=False, encoding="utf-8-sig")
    for lb, g in a.groupby("audit"):
        save([tile(crop_of(run, r.chain), f"E{r.event} {mmss(r.time_s)} {r.dir[0]}") for r in g.itertuples()], out, lb)
    # 미탐 후보: 이벤트 체인에 없고 x 이동 300px 이상, 같은 방향 이벤트와 2초 이상 떨어진 트랙
    tr = pd.read_csv(run / "tracks_split.csv"); pts = np.load(run / "pts.npy"); used = {int(x) for c in a.chain.astype(str) for x in c.split("|")}
    tr = tr.assign(bx=(tr.x1 + tr.x2) / 2, h=tr.y2 - tr.y1); cand = []
    for t, g in tr[~tr.tid.isin(used)].groupby("tid"):
        if len(g) < 5 or g.h.max() < 120 or np.ptp(g.bx) < 300: continue
        ds = "R2L" if g.bx.iloc[-1] < g.bx.iloc[0] else "L2R"; t0, t1 = pts[g.frame.iloc[0]], pts[g.frame.iloc[-1]]
        if ((a.dir == ds) & (a.time_s >= t0 - 2) & (a.time_s <= t1 + 2)).any(): continue
        cand.append((t0, t, ds, g.loc[g.h.idxmax()]))
    video = ROOT / "data/COUNT" / f"{v}.mp4"
    save([tile(frame_crop(video, pts[int(b.frame)], b), f"T{t} {mmss(t0)} {ds[0]}") for t0, t, ds, b in sorted(cand, key=lambda x: x[0])], out, "miss")
    print(v, a.audit.value_counts().to_dict(), "미탐 후보", len(cand), "시트", len(list(out.glob("*.jpg"))))
