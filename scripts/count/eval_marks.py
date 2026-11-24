"""표식 정답(사용자 박스 수정 240장) 저장·평가.

save: 박스 수정 페이지 db(edit/r2) json → data/COUNT/marks/gt_edit.csv (file,video,cls,x1,y1,x2,y2 / 박스 없는 크롭은 cls=-1 한 줄)
eval: 모델(.pt) 또는 라벨 폴더(<영상>.csv, keep2) 를 정답과 비교해 클래스별 P/R (IoU>=0.5, 작은 박스 고려해 0.3 도 같이)
사용: eval_marks.py save <edit json> | eval_marks.py eval <model.pt|labels_dir> [--videos ..] [--conf 0.25]
"""
import argparse, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"; GT = MK / "gt_edit.csv"; NAMES = ["plate", "beacon", "number"]


def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-9)


def score(gt, pred, th):
    S = {k: [0, 0, 0] for k in range(3)}  # tp fp fn
    for f in gt:
        for k in range(3):
            G = [b for c, *b in gt[f] if c == k]; D = [b for c, *b in pred.get(f, []) if c == k]; used = set()
            for d in D:
                m = [i for i, q in enumerate(G) if i not in used and iou(d, q) >= th]
                if m: used.add(m[0]); S[k][0] += 1
                else: S[k][1] += 1
            S[k][2] += len(G) - len(used)
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("src"); ap.add_argument("--videos", nargs="*"); ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if a.cmd == "save":
        e = json.load(open(a.src, encoding="utf-8")); e = e.get("m", e); rows = []
        for k, v in e.items():
            if not v.get("d"): continue
            vid, f = k.split("/")
            rows += [dict(file=f, video=vid, cls=b[0], x1=b[1], y1=b[2], x2=b[3], y2=b[4]) for b in v["b"]] or [dict(file=f, video=vid, cls=-1, x1=0, y1=0, x2=0, y2=0)]
        pd.DataFrame(rows).to_csv(GT, index=False); print("saved", GT, len(rows), "rows,", len({(r['video'], r['file']) for r in rows}), "crops"); raise SystemExit
    g = pd.read_csv(GT)
    if a.videos: g = g[g.video.isin(a.videos)]
    gt = {}
    for (v, f), x in g.groupby(["video", "file"]): gt[f"{v}/{f}"] = [(int(r.cls), r.x1, r.y1, r.x2, r.y2) for r in x.itertuples() if r.cls >= 0]
    pred = {}
    if a.src.endswith(".pt"):
        from ultralytics import YOLO
        m = YOLO(a.src); ks = list(gt)
        for s in range(0, len(ks), 32):
            R = m.predict([str(MK / "crops" / k) for k in ks[s:s + 32]], imgsz=640, conf=a.conf, verbose=False)
            for k, r in zip(ks[s:s + 32], R): pred[k] = [(int(c), *b) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
    else:
        for v in g.video.unique():
            d = pd.read_csv(Path(a.src) / f"{v}.csv"); d = d[d.keep2 == 1] if "keep2" in d else d
            for f, x in d.groupby("file"):
                if f"{v}/{f}" in gt: pred[f"{v}/{f}"] = [(int(r.cls), r.x1, r.y1, r.x2, r.y2) for r in x.itertuples()]
    print(f"{a.src}  crops {len(gt)}")
    for th in (0.5, 0.3):
        S = score(gt, pred, th)
        print(f"  IoU>={th}: " + "  ".join(f"{NAMES[k]} P {tp / max(1, tp + fp):.2f} R {tp / max(1, tp + fn):.2f} (gt {tp + fn})" for k, (tp, fp, fn) in S.items()))
