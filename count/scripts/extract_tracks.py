"""COUNT 1단계: 영상 → 차량 탐지+추적 → 트랙 CSV + 트랙 크롭 저장.

탐지/추적은 비싸니 한 번만 돌리고, 카운팅 로직·세이프티카 판별은 이 산출물로 오프라인 반복한다.

출력 (runs/count/<영상명>/):
  tracks.csv   frame,tid,cls,conf,x1,y1,x2,y2   (원본 해상도 좌표)
  crops/<tid>/<frame>.jpg   박스 높이 >= --crop-min-h 인 검출만, 트랙당 --crop-every 프레임 간격
  meta.json

사용:
  python count/scripts/extract_tracks.py data/COUNT/260701_s1_rtspcam.mp4 --model yolo26m.pt
"""
import argparse, csv, json, time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VEHICLE = [2, 5, 7]  # COCO car, bus, truck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="이 장면은 대상 차량이 커서 640 이 1280 보다 근접 차량 검출이 훨씬 안정적 (1280: conf 0.2~0.3 로 추락)")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=2, help="프레임 간격 (20fps 기준 2 → 10fps)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--crop-min-h", type=int, default=160)
    ap.add_argument("--crop-every", type=int, default=6, help="트랙당 크롭 저장 간격(처리 프레임 기준)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    vid = Path(a.video)
    out = Path(a.out) if a.out else ROOT / "runs" / "count" / vid.stem
    (out / "crops").mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)

    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    json.dump(dict(video=str(vid), fps=fps, frames=n, W=W, H=H, model=a.model, imgsz=a.imgsz,
                   conf=a.conf, stride=a.stride, tracker=a.tracker), open(out / "meta.json", "w"), indent=1)

    f = open(out / "tracks.csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["frame", "tid", "cls", "conf", "x1", "y1", "x2", "y2"])
    seen = {}  # tid -> 처리 카운터
    t0 = time.time(); fi = -1; proc = 0
    while True:
        ok = cap.grab(); fi += 1
        if not ok: break
        if fi % a.stride: continue
        ok, img = cap.retrieve()
        if not ok: break
        r = model.track(img, persist=True, imgsz=a.imgsz, conf=a.conf, classes=VEHICLE,
                        tracker=a.tracker, half=True, verbose=False)[0]
        proc += 1
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                     r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                w.writerow([fi, tid, c, f"{cf:.3f}", int(x1), int(y1), int(x2), int(y2)])
                if y2 - y1 >= a.crop_min_h:
                    k = seen.get(tid, 0); seen[tid] = k + 1
                    if k % a.crop_every == 0:
                        d = out / "crops" / str(tid); d.mkdir(exist_ok=True)
                        cv2.imwrite(str(d / f"{fi:06d}.jpg"), img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)],
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if proc % 500 == 0:
            el = time.time() - t0
            print(f"[{vid.stem}] {fi}/{n} {proc/el:.1f} it/s", flush=True)
        if a.max_frames and fi >= a.max_frames: break
    f.close()
    print(f"done {vid.stem}: {proc} frames in {time.time()-t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
