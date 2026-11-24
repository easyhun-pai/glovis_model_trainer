"""실험: frozen 백본 임베딩 + 로지스틱 회귀 → 이벤트(트랙) 단위 '비화물(세이프티/작업차)' 판별.

평가: leave-one-video-out. 크롭 확률을 트랙별로 집계(top-k 평균) → 이벤트 판정.
비화물 = label != cargo (세이프티카 + 작업차). 카운트 = 화물로 판정된 이벤트 수.

사용: python count/scripts/exp_embed_cls.py --model facebook/dinov2-base
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def embed(paths, model_id, bs=32, cache=None, top=1.0):
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == list(paths): return d["X"]
    proc = AutoImageProcessor.from_pretrained(model_id)
    m = AutoModel.from_pretrained(model_id).cuda().eval().half()
    X = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(ROOT / p).convert("RGB") for p in paths[i:i + bs]]
        if top < 1.0: ims = [im.crop((0, 0, im.width, max(8, int(im.height * top)))) for im in ims]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        o = m(pixel_values=x)
        h = o.last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)  # CLS + 패치 평균
        X.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
    X = np.concatenate(X)
    if cache: np.savez(cache, X=X, paths=np.array(paths))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--top", type=float, default=1.0, help="크롭 상단 비율만 사용 (지붕/경광등 영역)")
    a = ap.parse_args()
    ix = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "runs/count/crops_ds").glob("*/index.csv"))], ignore_index=True)
    ix["y"] = (ix.label != "cargo").astype(int)
    cache = ROOT / "runs/count" / f"emb_{a.model.replace('/', '_')}_top{a.top}.npz"
    X = embed(list(ix.path), a.model, cache=cache, top=a.top)
    tot = dict(tp=0, fp=0, fn=0, tn=0); rows = []
    for v in ix.video.unique():
        tr, te = ix.video != v, ix.video == v
        if ix[tr].y.nunique() < 2: print(f"skip {v}: train 에 양/음 둘 다 없음"); continue
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[tr], ix.y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        t = ix[te].assign(p=p)
        ev = t.groupby("event").agg(y=("y", "first"), label=("label", "first"),
                                    p=("p", lambda s: np.sort(s.values)[-a.topk:].mean())).reset_index()
        ev["pred"] = (ev.p >= a.thr).astype(int)
        for _, r in ev.iterrows():
            k = ("tp" if r.pred else "fn") if r.y else ("fp" if r.pred else "tn"); tot[k] += 1
        gt_cnt = int((ev.y == 0).sum()); pr_cnt = int((ev.pred == 0).sum())
        bad = ev[ev.pred != ev.y]
        print(f"{v}: events={len(ev)} cargo GT={gt_cnt} pred={pr_cnt}  오판={len(bad)} "
              + " ".join(f"E{int(r.event)}({r.label},{r.p:.2f})" for _, r in bad.iterrows()))
        rows.append(ev.assign(video=v))
    print("total", tot)
    pd.concat(rows).to_csv(ROOT / "runs/count" / f"exp_embed_{a.model.replace('/', '_')}.csv", index=False)


if __name__ == "__main__":
    main()
