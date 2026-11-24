"""표식 초안 2판: 참조 크롭 여러 장(흰 경광등·측면 창문 숫자 포함)의 시각 프롬프트 임베딩을 클래스별로 평균해 경광등·번호를 다시 검출.

1판(draft_marks.py)은 참조 1장(빨강 경광등, 앞유리 숫자)뿐이라 흰 경광등·측면 큰 숫자를 놓쳤다(표식 검수 r1: 세이프티 크롭 75% 미탐).
출력: data/COUNT/marks/draft_vp2/<영상>.csv  (file,id,label,cls,conf,x1,y1,x2,y2,W,H)  cls 1 beacon, 2 number
사용: draft_marks2.py [--conf 0.1] 영상...
"""
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

ROOT = Path(__file__).resolve().parents[2]; M = ROOT / "_model_/count/yoloe-11l-seg.pt"; C = ROOT / "data/COUNT/marks/crops"
# (크롭, 클래스, 박스) — 1 beacon, 2 number.  좌표는 크롭 원본 픽셀.
REFS = [
    ("260818_s1_rtspcam/33_2458.jpg", 1, [[290, 158, 330, 205], [527, 150, 565, 192]]),   # 빨강 경광등
    ("260818_s1_rtspcam/33_2458.jpg", 2, [[328, 215, 375, 250], [68, 222, 100, 268]]),    # 앞유리·옆유리 2
    ("260731_rtspcam/159_16220.jpg", 1, [[99, 106, 123, 131], [231, 80, 255, 104]]),      # 흰 경광등
    ("260818_s2_rtspcam/87_10821.jpg", 2, [[480, 150, 504, 185]]),                        # 측면 뒷창 7
    ("260818_s2_rtspcam/531_53748.jpg", 2, [[413, 114, 438, 144], [472, 98, 494, 113]]),  # 측면 4, 뒷유리 4
]


def class_vpe(model, img, boxes):
    p = YOLOEVPSegPredictor(overrides=dict(task="segment", mode="predict", save=False, verbose=False, batch=1, imgsz=640))
    model.model.model[-1].nc = 1; model.model.names = ["object0"]
    p.set_prompts(dict(bboxes=np.array(boxes, float), cls=np.zeros(len(boxes), int))); p.setup_model(model=model.model)
    return torch.nn.functional.normalize(p.get_vpe(img)[0, 0], dim=-1)


def build(model):
    by = {1: [], 2: []}
    for f, k, b in REFS: by[k].append(class_vpe(model, cv2.imread(str(C / f)), b))
    v = torch.stack([torch.nn.functional.normalize(torch.stack(by[k]).mean(0), dim=-1) for k in (1, 2)])[None]
    model.model.set_classes(["beacon", "number"], v); model.predictor = None; model.task = "segment"
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--conf", type=float, default=0.1); a = ap.parse_args()
    m = build(YOLOE(M)); out = ROOT / "data/COUNT/marks/draft_vp2"; out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        d = pd.read_csv(C / f"{v}.csv"); rows = []
        for s in range(0, len(d), 16):
            b = d.iloc[s:s + 16]; R = m.predict([str(C / v / f) for f in b.file], conf=a.conf, verbose=False)
            for r, res in zip(b.itertuples(), R):
                H, W = res.orig_shape
                for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                    rows.append(dict(file=r.file, id=r.id, label=r.label, cls=int(k) + 1, conf=round(c, 3), x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2), W=W, H=H))
            if s % 1600 == 0: print(v, s, len(d), flush=True)
        df = pd.DataFrame(rows); df.to_csv(out / f"{v}.csv", index=False); print(v, df.groupby(["label", "cls"]).size().to_dict(), flush=True)
