"""realtime_sim.py 결과(events.csv)를 정답과 비교.

  gt_v2 (260701): python count/scripts/eval_rt.py runs/count/rt/<영상>_r3 --gt data/COUNT/gt_v2/<영상>.csv [--reviewed-only]
      time(mm:ss, 1초 단위) → 초+0.5 기준 ±tol 초, 같은 방향으로 탐욕 매칭. --reviewed-only 면 reviewed=1 인 마지막 행 시각까지만 비교.
  gt_seq (홀드아웃): python count/scripts/eval_rt.py runs/count/rt/<영상>_r3 --seq data/COUNT/gt_seq/<영상>.txt
      통과 순서 문자열(0=하역 화물 L2R, 1=선적 화물 R2L, 2=세이프티·제외, 방향 미구분)과 예측 순서를 전역 정렬(Needleman-Wunsch, 비용 1/1/1).
출력: <run>/eval_gt.json (+ seq 는 eval_seq.csv: 정렬 결과 한 줄씩)
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def mmss(x):
    m, s = str(x).split(":"); return int(m) * 60 + int(s)


def eval_gt(ev, gt_path, reviewed_only, tol):
    g = pd.read_csv(gt_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    g["t"] = g.time.map(mmss) + 0.5
    t_end = None
    if reviewed_only and "reviewed" in g:
        t_end = float(g[g.reviewed == "1"].t.max()) + 1.5
        g = g[g.reviewed == "1"]; ev = ev[ev.cross_s <= t_end]
    g = g.reset_index(drop=True); g["gdec"] = np.where(g.label == "cargo", "cargo", "exclude")
    used = set(); m = {}
    for r in ev.itertuples():
        c = g[(g.dir == r.dir) & (~g.index.isin(used)) & ((g.t - r.cross_s).abs() <= tol)]
        if len(c):
            j = (c.t - r.cross_s).abs().idxmin(); used.add(j); m[r.Index] = j
    ev = ev.assign(gt_label=[g.label[m[i]] if i in m else "" for i in ev.index], gt_time=[g.time[m[i]] if i in m else "" for i in ev.index])
    res = dict(t_end=t_end, gt_events=len(g), pred_events=len(ev), matched=len(m))
    for op in ("load", "unload"):
        res[f"{op}_cargo_gt"] = int(((g.op == op) & (g.gdec == "cargo")).sum()); res[f"{op}_cargo_pred"] = int(((ev.op == op) & (ev.decision == "cargo")).sum())
    res["exclude_gt"] = int((g.gdec == "exclude").sum()); res["exclude_pred"] = int((ev.decision == "exclude").sum())
    mm = ev[ev.gt_label != ""]; gl = np.where(mm.gt_label == "cargo", "cargo", "exclude")
    res.update(false=[f"{r.cross_time} {r.dir} {r.decision}" for r in ev[ev.gt_label == ""].itertuples()],
               missed=[f"{r.time} {r.dir} {r.label}" for r in g[~g.index.isin(used)].itertuples()],
               decision_wrong=[f"{r.cross_time} {r.dir} pred={r.decision} gt={r.gt_label} ({r.reason})" for r, x in zip(mm.itertuples(), gl) if x != r.decision])
    return res


def nw(a, b):
    """전역 정렬. 반환: [(op, i, j)] op ∈ match/sub/ins(예측에만)/del(정답에만)."""
    n, m = len(a), len(b); D = np.zeros((n + 1, m + 1), int); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (a[i - 1] != b[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out.append(("match" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1)); i -= 1; j -= 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out.append(("ins", i - 1, None)); i -= 1
        else:
            out.append(("del", None, j - 1)); j -= 1
    return out[::-1], int(D[n, m])


def eval_seq(ev, seq_path):
    gt = "".join(l.strip() for l in open(seq_path, encoding="utf-8") if l.strip() and not l.startswith("#"))
    ev = ev.sort_values("cross_s").reset_index(drop=True)
    pr = "".join("2" if d == "exclude" else ("1" if o == "load" else "0") for d, o in zip(ev.decision, ev.op))
    ops, dist = nw(pr, gt)
    name = {"0": "하역화물", "1": "선적화물", "2": "제외(세이프티)"}
    rows = []
    for op, i, j in ops:
        rows.append(dict(op=op, gt_pos=j, gt=name[gt[j]] if j is not None else "", pred_time=ev.cross_time[i] if i is not None else "",
                         pred_s=ev.cross_s[i] if i is not None else None, pred=name[pr[i]] if i is not None else "",
                         dir=ev.dir[i] if i is not None else "", reason=ev.reason[i] if i is not None else ""))
    R = pd.DataFrame(rows)
    # 정답에만 있는(미탐) 항목은 앞뒤 예측 시각 사이로 위치를 알려준다
    for k in R.index[R.op == "del"]:
        prev = R.pred_time[:k][R.pred_time[:k] != ""]; nxt = R.pred_time[k + 1:][R.pred_time[k + 1:] != ""]
        R.loc[k, "pred_time"] = f"({prev.iloc[-1] if len(prev) else '시작'} ~ {nxt.iloc[0] if len(nxt) else '끝'} 사이)"
    cnt = lambda s: {"unload": s.count("0"), "load": s.count("1"), "exclude": s.count("2")}
    return R, dict(gt_len=len(gt), pred_len=len(pr), edit_distance=dist, gt_counts=cnt(gt), pred_counts=cnt(pr),
                   n_match=int((R.op == "match").sum()), n_sub=int((R.op == "sub").sum()), n_ins=int((R.op == "ins").sum()), n_del=int((R.op == "del").sum()),
                   errors=[f"{r.op}: 정답#{'' if r.gt_pos is None or pd.isna(r.gt_pos) else int(r.gt_pos)} {r.gt or '-'} / 예측 {r.pred or '-'} {r.pred_time} {r.dir}"
                           for r in R[R.op != "match"].itertuples()])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--gt"); ap.add_argument("--seq")
    ap.add_argument("--reviewed-only", action="store_true"); ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args(); run = Path(a.run)
    ev = pd.read_csv(run / "events.csv", encoding="utf-8-sig")
    if a.gt: res = eval_gt(ev, a.gt, a.reviewed_only, a.tol)
    else:
        R, res = eval_seq(ev, a.seq); R.to_csv(run / "eval_seq.csv", index=False, encoding="utf-8-sig")
    json.dump(res, open(run / "eval_gt.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
