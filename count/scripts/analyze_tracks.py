"""COUNT 2단계(탐색): tracks.csv → 트랙 요약 + 궤적 오버레이 + 이동 트랙 이벤트 시트.

출력 (runs/count/<영상명>/):
  track_summary.csv   tid별 시작/끝 프레임, 이동량, 최대 박스 높이, 시작/끝 bottom-center
  traj.jpg            이동 트랙 궤적을 대표 프레임 위에 표시
  events.jpg          이동 트랙별 대표 크롭 시트 (tid, 시각)
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def summarize(df):
    df = df.copy()
    df["bx"] = (df.x1 + df.x2) / 2; df["by"] = df.y2; df["h"] = df.y2 - df.y1
    rows = []
    for tid, g in df.groupby("tid"):
        g = g.sort_values("frame")
        pts = g[["bx", "by"]].values
        path = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() if len(g) > 1 else 0
        rows.append(dict(tid=tid, f0=g.frame.iloc[0], f1=g.frame.iloc[-1], n=len(g),
                         x0=pts[0, 0], y0=pts[0, 1], x1=pts[-1, 0], y1=pts[-1, 1],
                         disp=float(np.linalg.norm(pts[-1] - pts[0])), path=float(path),
                         spread=float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])),
                         hmax=g.h.max(), conf=g.conf.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/count/<영상명>")
    ap.add_argument("--min-spread", type=float, default=250, help="이동 트랙 판정: bottom-center x/y 범위 합(px)")
    ap.add_argument("--min-h", type=float, default=150)
    a = ap.parse_args()
    run = Path(a.run); meta = json.load(open(run / "meta.json"))
    df = pd.read_csv(run / "tracks.csv")
    s = summarize(df); s.to_csv(run / "track_summary.csv", index=False)
    mv = s[(s.spread >= a.min_spread) & (s.hmax >= a.min_h)].sort_values("f0")
    print(f"{run.name}: tracks={len(s)} moving={len(mv)}")

    cap = cv2.VideoCapture(meta["video"]); cap.set(1, 10); _, bg = cap.read()
    rng = np.random.default_rng(0)
    for _, r in mv.iterrows():
        g = df[df.tid == r.tid].sort_values("frame")
        pts = np.stack([(g.x1 + g.x2) / 2, g.y2], 1).astype(np.int32)
        col = tuple(int(v) for v in rng.integers(60, 255, 3))
        cv2.polylines(bg, [pts], False, col, 3)
        cv2.circle(bg, tuple(pts[0]), 8, (0, 255, 0), -1); cv2.circle(bg, tuple(pts[-1]), 8, (0, 0, 255), -1)
        cv2.putText(bg, str(r.tid), tuple(pts[len(pts) // 2]), 0, 1.2, col, 3)
    cv2.imwrite(str(run / "traj.jpg"), bg)

    tiles = []
    for _, r in mv.iterrows():
        d = run / "crops" / str(int(r.tid)); fs = sorted(d.glob("*.jpg")) if d.exists() else []
        if fs:
            im = cv2.imread(str(fs[len(fs) // 2]))
        else:
            im = np.zeros((10, 10, 3), np.uint8)
        im = cv2.resize(im, (320, 200))
        t = r.f0 / meta["fps"]
        cv2.rectangle(im, (0, 0), (320, 28), (0, 0, 0), -1)
        cv2.putText(im, f"#{int(r.tid)} {int(t//60)}:{int(t%60):02d} n{int(r.n)}", (4, 21), 0, 0.7, (0, 255, 255), 2)
        tiles.append(im)
    if tiles:
        cols = 8; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(run / "events.jpg"), sheet)


if __name__ == "__main__":
    main()
