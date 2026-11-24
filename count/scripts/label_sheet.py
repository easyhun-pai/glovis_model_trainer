"""GT 라벨링 보조: 이벤트별 트랙 크롭 3장(앞/중/뒤)을 한 줄로, 여러 이벤트를 한 장에.

extract_tracks 의 crops/<tid>/ 를 사용. 출력: runs/count/<영상>/label_sheet_<k>.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--dir", default=None, help="R2L/L2R 필터")
    a = ap.parse_args()
    run = Path(a.run); ev = pd.read_csv(run / "events.csv")
    if a.dir: ev = ev[ev.dir == a.dir]
    rows = []
    for _, r in ev.iterrows():
        fs = sorted((run / "crops" / str(int(r.tid))).glob("*.jpg"))
        pick = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, 3)] if fs else []
        ims = [cv2.resize(cv2.imread(str(p)), (420, 260)) for p in pick] + [np.zeros((260, 420, 3), np.uint8)] * (3 - len(pick))
        row = np.hstack(ims)
        cv2.rectangle(row, (0, 0), (420, 30), (0, 0, 0), -1)
        cv2.putText(row, f"E{int(r.event)} {r.time_s}s {r.dir}", (4, 23), 0, 0.75, (0, 255, 255), 2)
        rows.append(row)
    for k in range(0, len(rows), a.per):
        rr = rows[k:k + a.per]
        if len(rr) % 2: rr.append(np.zeros_like(rr[0]))
        cv2.imwrite(str(run / f"label_sheet_{k // a.per}.jpg"),
                    np.vstack([np.hstack(rr[i:i + 2]) for i in range(0, len(rr), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(rows)} events → {(len(rows) + a.per - 1) // a.per} sheets")


if __name__ == "__main__":
    main()
