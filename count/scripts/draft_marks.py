"""2차(표식) 모델 라벨 초안: 크롭마다 번호판(YOLOE 텍스트), 경광등·창문 번호(YOLOE 시각 프롬프트) 검출.

클래스: 0 plate, 1 beacon, 2 number.  크롭은 위쪽 30% 여유가 있어 차량 박스 윗변이 크롭 높이의 약 23% 지점.
위치 규칙: beacon 중심 y < 0.40H, number 중심 0.15H < y < 0.60H, plate 가로/세로 >= 1.5.
화물차 크롭은 검출 결과를 det 에만 남기고(오검출 통계) 라벨은 빈 파일(배경)로 둔다.
출력: data/COUNT/marks/draft/<영상>.csv (file,label,cls,conf,x1,y1,x2,y2,keep)
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

ROOT = Path(__file__).resolve().parents[2]; M = ROOT / "_model_/count/yoloe-11l-seg.pt"; C = ROOT / "data/COUNT/marks/crops"
REF = C / "260818_s1_rtspcam/33_2458.jpg"  # 세이프티 2번: 경광등 2, 번호 2 (앞유리·옆유리)
VP = dict(bboxes=np.array([[290, 158, 330, 205], [527, 150, 565, 192], [328, 215, 375, 250], [68, 222, 100, 268]], float), cls=np.array([0, 0, 1, 1]))
TH = {0: 0.25, 1: 0.15, 2: 0.35}
tp = YOLOE(M); tp.set_classes(["license plate"], tp.get_text_pe(["license plate"])); vp = YOLOE(M)
out = ROOT / "data/COUNT/marks/draft"; out.mkdir(parents=True, exist_ok=True)
for v in sys.argv[1:]:
    d = pd.read_csv(C / f"{v}.csv"); rows = []
    for s in range(0, len(d), 16):
        b = d.iloc[s:s + 16]; ims = [cv2.imread(str(C / v / f)) for f in b.file]
        R1 = tp.predict(ims, conf=0.1, verbose=False)
        R2 = vp.predict(ims, refer_image=str(REF), visual_prompts=VP, predictor=YOLOEVPSegPredictor, conf=0.1, verbose=False)
        for r, im, r1, r2 in zip(b.itertuples(), ims, R1, R2):
            H, W = im.shape[:2]
            for res, off in ((r1, 0), (r2, 1)):
                for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                    k = int(k) + off; cy = (y1 + y2) / 2 / H; w, h = x2 - x1, y2 - y1
                    ok = c >= TH[k] and (w * h) < 0.05 * W * H
                    if k == 0: ok &= w >= 1.5 * h
                    if k == 1: ok &= cy < 0.40
                    if k == 2: ok &= 0.15 < cy < 0.60
                    rows.append(dict(file=r.file, id=r.id, label=r.label, cls=k, conf=round(c, 3), x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2), W=W, H=H, keep=int(ok)))
        if s % 800 == 0: print(v, s, len(d), flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / f"{v}.csv", index=False)
    k = df[df.keep == 1]; print(v, "kept by label/cls:", k.groupby(["label", "cls"]).size().to_dict(), flush=True)
