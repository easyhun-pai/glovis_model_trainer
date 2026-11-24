"""COUNT 검수: 이벤트별 통과 순간 프레임 시트 + 탐지 독립 '라인 점유' 교차검증.

1) review_sheet_*.jpg : 이벤트마다 통과 프레임(박스 표시) 타일 → 사람이 화물/세이프티/기타 판정
2) occupancy.csv / occupancy.png : 라인 주변 세로 띠의 배경차분 점유율 시계열.
   탐지와 무관한 신호라, 점유 구간인데 이벤트가 없으면 '미탐 후보'로 출력.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_frames(video, want):
    """순차 디코딩으로 정확한 프레임 획득 (cap.set 시크는 H.264 에서 키프레임 근처로 어긋남)."""
    cap = cv2.VideoCapture(video); want = set(want); out = {}; fi = -1; last = max(want)
    while fi < last:
        if not cap.grab(): break
        fi += 1
        if fi in want: out[fi] = cap.retrieve()[1]
    return out


def sheet(run, meta, ev, tr, cols=4, per=24):
    frames = read_frames(meta["video"], sorted(set(ev.frame.astype(int)))); tiles = []
    for _, r in ev.iterrows():
        im = frames.get(int(r.frame))
        if im is None: continue
        g = tr[(tr.frame == r.frame)]
        for _, b in g.iterrows():
            col = (0, 255, 0) if b.tid == r.tid else (128, 128, 128)
            cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, 4)
        x0 = 0; im = im[500:1440, x0:x0 + 2560]
        im = cv2.resize(im, (800, 294))
        cv2.rectangle(im, (0, 0), (800, 34), (0, 0, 0), -1)
        cv2.putText(im, f"E{int(r.event)} tid{int(r.tid)} t={r.time_s}s {r.dir}", (6, 26), 0, 0.9, (0, 255, 255), 2)
        tiles.append(im)
    for k in range(0, len(tiles), per):
        t = tiles[k:k + per]; t += [np.zeros_like(t[0])] * (-len(t) % cols)
        cv2.imwrite(str(run / f"review_sheet_{k // per}.jpg"),
                    np.vstack([np.hstack(t[i:i + cols]) for i in range(0, len(t), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def occupancy(meta, line_x, y_lo, y_hi, step=2, half=40):
    cap = cv2.VideoCapture(meta["video"])
    bg = None; out = []; fi = -1
    while True:
        ok = cap.grab(); fi += 1
        if not ok: break
        if fi % step: continue
        _, im = cap.retrieve()
        roi = cv2.cvtColor(im[int(y_lo):int(y_hi), int(line_x - half):int(line_x + half)], cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (5, 5), 0).astype(np.float32)
        if bg is None: bg = roi.copy()
        d = np.abs(roi - bg) > 25
        out.append((fi, d.mean()))
        # 점유가 낮을 때만 배경 갱신 (조명 변화 추종)
        if d.mean() < 0.05: cv2.accumulateWeighted(roi, bg, 0.05)
    return pd.DataFrame(out, columns=["frame", "occ"])


def episodes(occ, fps, thr=0.25, min_len=0.3, merge=0.6):
    on = occ.occ.values > thr; f = occ.frame.values; eps = []; s = None
    for i, v in enumerate(on):
        if v and s is None: s = i
        if not v and s is not None: eps.append([f[s], f[i - 1]]); s = None
    if s is not None: eps.append([f[s], f[-1]])
    m = []
    for a, b in eps:
        if m and (a - m[-1][1]) / fps < merge: m[-1][1] = b
        else: m.append([a, b])
    return [(a, b) for a, b in m if (b - a) / fps >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--line-x", type=float, default=1000)
    ap.add_argument("--y-band", type=float, nargs=2, default=[950, 1300])
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    run = Path(a.run); meta = json.load(open(run / "meta.json"))
    ev = pd.read_csv(run / "events.csv"); tr = pd.read_csv(run / "tracks.csv")
    if not a.no_sheet: sheet(run, meta, ev, tr)
    occ = occupancy(meta, a.line_x, *a.y_band); occ.to_csv(run / "occupancy.csv", index=False)
    eps = episodes(occ, meta["fps"])
    fps = meta["fps"]; miss = []
    for s, e in eps:
        hit = ev[(ev.frame >= s - fps) & (ev.frame <= e + fps)]
        if hit.empty: miss.append((s, e))
    print(f"{run.name}: events={len(ev)} occ_episodes={len(eps)} episodes_without_event={len(miss)}")
    for s, e in miss: print(f"  미탐후보 {s/fps:.1f}s ~ {e/fps:.1f}s (frame {s}-{e})")
    orphan = [r.event for _, r in ev.iterrows() if not any(s - fps <= r.frame <= e + fps for s, e in eps)]
    if orphan: print(f"  점유 없이 생긴 이벤트(오검 후보): {orphan}")


if __name__ == "__main__":
    main()
