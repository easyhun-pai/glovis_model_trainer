"""실험: zero-shot 번호판 검출(YOLOE 텍스트 프롬프트) → 트랙 단위 '번호판 있음 = 비화물' 판정.

근거: 선적/하역 화물차는 번호판이 없고, 세이프티카·작업차는 번호판이 있다.
학습 없이 동작하므로 baseline 겸 자동 라벨 소스로 쓴다.

출력: runs/count/exp_plate_zeroshot.csv (이벤트별 plate_hits, max_conf, 판정)
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "_model_/count/yoloe-11l-seg.pt"))
    ap.add_argument("--prompt", default="license plate")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--min-hits", type=int, default=1)
    a = ap.parse_args()
    m = YOLOE(a.model); names = [a.prompt]; m.set_classes(names, m.get_text_pe(names))
    ix = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "runs/count/crops_ds").glob("*/index.csv"))], ignore_index=True)
    hits, mc = [], []
    for i in range(0, len(ix), 32):
        res = m.predict([str(ROOT / p) for p in ix.path[i:i + 32]], conf=a.conf, verbose=False)
        for r in res:
            hits.append(len(r.boxes)); mc.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    ix["hits"] = [h > 0 for h in hits]; ix["pconf"] = mc
    ev = ix.groupby(["video", "event"]).agg(label=("label", "first"), n=("path", "size"),
                                             plate_frames=("hits", "sum"), max_conf=("pconf", "max")).reset_index()
    ev["pred_noncargo"] = ev.plate_frames >= a.min_hits
    ev["y"] = ev.label != "cargo"
    ev.to_csv(ROOT / "runs/count/exp_plate_zeroshot.csv", index=False)
    for v, g in ev.groupby("video"):
        bad = g[g.pred_noncargo != g.y]
        print(f"{v}: events={len(g)} cargo GT={int((~g.y).sum())} pred={int((~g.pred_noncargo).sum())} 오판={len(bad)} "
              + " ".join(f"E{r.event}({r.label},{r.plate_frames}/{r.n},{r.max_conf:.2f})" for r in bad.itertuples()))
    print(ev.groupby("label")[["plate_frames", "max_conf"]].describe().round(2).T)


if __name__ == "__main__":
    main()
