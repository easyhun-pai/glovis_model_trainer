"""표식 박스 수정 페이지 데이터: 1차 검수(marks/r1)한 세이프티·작업 크롭 240장에 draft_r2 박스를 미리 채워 내보낸다.
사용자가 박스를 그리고 지워 정답(평가·학습용)을 만든다.  db 문서 edit/r2 의 m 맵 {영상/파일: {b:[[cls,x1,y1,x2,y2]..], d:1}}.
출력: runs/count/review/marks_edit/{items.json, img/NNN.jpg}
사용: build_marks_edit.py <marks/r1 json> [--src draft_r2]
"""
import argparse, json
from pathlib import Path
import cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"; OUT = ROOT / "runs/count/review/marks_edit"
ap = argparse.ArgumentParser(); ap.add_argument("review"); ap.add_argument("--src", default="draft_r2"); ap.add_argument("--maxw", type=int, default=820); a = ap.parse_args()
rv = json.load(open(a.review, encoding="utf-8")); rv = rv.get("m", rv)
old = json.load(open(ROOT / "runs/count/review/marks/items.json", encoding="utf-8")); old = old["items"] if isinstance(old, dict) else old
sv = [x for x in old if x["t"] == "sv"]
order = {"both": 0, "miss": 1, "wrong": 2, "ok": 3}; sv.sort(key=lambda x: order.get(rv.get(x["k"]), 4))
(OUT / "img").mkdir(parents=True, exist_ok=True); D = {}; P = {}


def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-9)
items = []
for i, x in enumerate(sv):
    v, f = x["k"].split("/")
    if v not in D: D[v] = {k: g for k, g in pd.read_csv(MK / a.src / f"{v}.csv").groupby("file")}
    im = cv2.imread(str(MK / "crops" / v / f)); H, W = im.shape[:2]; s = min(1, a.maxw / W)
    if s < 1: im = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "img" / f"{i:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 86])
    g = D[v].get(f); b = [] if g is None else [[int(r.cls), int(r.x1), int(r.y1), int(r.x2), int(r.y2)] for r in g.itertuples()]
    # 번호판 후보 보강: 1판 초안(YOLOE 텍스트) conf>=0.15, 차량 박스 안, 기존 번호판과 겹치지 않는 것
    if v not in P: P[v] = (pd.read_csv(MK / "draft" / f"{v}.csv").query("cls == 0 and conf >= 0.15").groupby("file"), pd.read_csv(MK / "crops" / f"{v}.csv").set_index("file"))
    pg, cc = P[v]; q = cc.loc[f]; X1, Y1, X2, Y2 = q.bx1 - q.cx1, q.by1 - q.cy1, q.bx2 - q.cx1, q.by2 - q.cy1
    if f in pg.groups:
        for r in pg.get_group(f).itertuples():
            mx, my = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
            if not (X1 <= mx <= X2 and Y1 <= my <= Y2) or (r.x2 - r.x1) < 1.5 * (r.y2 - r.y1): continue
            if any(o[0] == 0 and iou(o[1:], (r.x1, r.y1, r.x2, r.y2)) > 0.3 for o in b): continue
            b.append([0, int(r.x1), int(r.y1), int(r.x2), int(r.y2)])
    items.append(dict(k=x["k"], img=f"img/{i:03d}.jpg", w=W, h=H, n=x["n"], r=rv.get(x["k"], ""), b=b))
json.dump(dict(items=items), open(OUT / "items.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(items), "items", sum(len(x["b"]) for x in items), "boxes")
