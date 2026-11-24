"""draft_marks 결과에 소유 위치 규칙 적용 → keep2.
beacon: 중심 x 가 차량 박스 x 범위 안, 중심 y 가 [박스 윗변-0.25h, 윗변+0.20h]
number/plate: 중심이 차량 박스 안 (number 는 박스 위쪽 60% 안)
사용: filter_marks.py <영상>...
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; NUM_TH = 0.2
for v in sys.argv[1:]:
    d = pd.read_csv(ROOT / "data/COUNT/marks/draft" / f"{v}.csv"); c = pd.read_csv(ROOT / "data/COUNT/marks/crops" / f"{v}.csv").set_index("file")
    c = c.assign(X1=c.bx1 - c.cx1, Y1=c.by1 - c.cy1, X2=c.bx2 - c.cx1, Y2=c.by2 - c.cy1)[["X1", "Y1", "X2", "Y2"]]
    d = d.drop(columns=[x for x in ("X1", "Y1", "X2", "Y2", "keep2") if x in d]).join(c, on="file")
    mx, my, h = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, d.Y2 - d.Y1
    inx = (mx >= d.X1) & (mx <= d.X2)
    g = ((d.cls == 1) & inx & (my >= d.Y1 - 0.25 * h) & (my <= d.Y1 + 0.20 * h)) | \
        ((d.cls == 2) & inx & (my >= d.Y1) & (my <= d.Y1 + 0.6 * h)) | ((d.cls == 0) & inx & (my >= d.Y1) & (my <= d.Y2))
    # number 는 시각 프롬프트 점수가 낮게 나와 기준을 0.2 로 (차량 박스 안 창문 높이로 위치를 묶었으므로)
    base = (d.keep == 1) | ((d.cls == 2) & (d.conf >= NUM_TH) & ((d.x2 - d.x1) * (d.y2 - d.y1) < 0.05 * d.W * d.H))
    d["keep2"] = (base & g).astype(int); d.to_csv(ROOT / "data/COUNT/marks/draft" / f"{v}.csv", index=False)
    k = d[d.keep2 == 1]; n = d.groupby("label").file.nunique()
    per = k.groupby(["label", "cls"]).file.nunique().unstack(fill_value=0)
    print(v, "crops", n.to_dict()); print((per.T / n).T.round(3).rename(columns={0: "plate", 1: "beacon", 2: "number"}))
