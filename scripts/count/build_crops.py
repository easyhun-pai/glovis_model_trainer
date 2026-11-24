"""GT 라벨된 이벤트 → 트랙 단위 크롭 데이터셋 (세이프티카/작업차 vs 화물 분류 실험용).

events.csv 의 각 이벤트를 GT(data/COUNT/gt/<영상>.csv, time_s 매칭)와 붙이고,
해당 트랙의 모든 검출 박스를 --every 프레임 간격으로 순차 디코딩해 크롭 저장.

출력: runs/count/crops_ds/<영상>/<label>/e<event>_t<tid>_f<frame>.jpg
      runs/count/crops_ds/<영상>/index.csv   (path, video, event, tid, frame, label, safety_no)
"""
import argparse, json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=150)
    ap.add_argument("--pad", type=float, default=0.08, help="박스 위쪽 여유(경광등이 박스 밖으로 잘리는 것 방지)")
    a = ap.parse_args()
    run = Path(a.run); meta = json.load(open(run / "meta.json"))
    gt = pd.read_csv(a.gt or ROOT / "data" / "COUNT" / "gt" / f"{run.name}.csv")
    ev = pd.read_csv(run / "events.csv"); tr = pd.read_csv(run / "tracks.csv")
    # time 매칭 (1초 이내)
    rows = []
    for _, g in gt.iterrows():
        d = (ev.time_s - g.time_s).abs()
        if d.min() > 1.0: print(f"[!] GT {g.time_s}s 에 맞는 이벤트 없음"); continue
        e = ev.loc[d.idxmin()]; rows.append(dict(event=int(e.event), tid=int(e.tid), label=g.label, safety_no=g.safety_no))
    lab = pd.DataFrame(rows)
    sel = tr[tr.tid.isin(lab.tid) & (tr.y2 - tr.y1 >= a.min_h)].merge(lab, on="tid")
    sel = sel[sel.groupby("tid").cumcount() % a.every == 0]
    want = sel.groupby("frame")
    out = ROOT / "runs" / "count" / "crops_ds" / run.name
    cap = cv2.VideoCapture(meta["video"]); fi = -1; last = sel.frame.max(); idx = []
    frames = set(sel.frame)
    while fi < last:
        if not cap.grab(): break
        fi += 1
        if fi not in frames: continue
        im = cap.retrieve()[1]; H, W = im.shape[:2]
        for b in want.get_group(fi).itertuples():
            h = b.y2 - b.y1; y1 = max(0, int(b.y1 - a.pad * h))
            crop = im[y1:int(b.y2), max(0, int(b.x1)):min(W, int(b.x2))]
            d = out / b.label; d.mkdir(parents=True, exist_ok=True)
            p = d / f"e{b.event}_t{b.tid}_f{fi}.jpg"; cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            idx.append(dict(path=str(p.relative_to(ROOT)), video=run.name, event=b.event, tid=b.tid, frame=fi,
                            label=b.label, safety_no=b.safety_no))
    pd.DataFrame(idx).to_csv(out / "index.csv", index=False)
    print(f"{run.name}: {len(idx)} crops  " + str(pd.DataFrame(idx).groupby('label').event.nunique().to_dict()) + " events")


if __name__ == "__main__":
    main()
