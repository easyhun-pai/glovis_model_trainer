"""COUNT 평가: events.csv(+선택: 비화물 판정) vs GT(data/COUNT/gt_v2/<영상>.csv).

매칭: 같은 방향, |Δt| <= --tol 초, 1:1 그리디.
출력: 영상별 이벤트 검출 (미탐/오검) + 화물 카운트(방향별) GT vs 예측.
  --pred <csv>: event, noncargo(0/1) 컬럼을 가진 판정 파일 (없으면 이벤트 검출만 평가)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]


def to_sec(t):
    m, c = str(t).split(":"); return int(m) * 60 + int(c)


def match(ev, gt, tol):
    """방향별 시간차 합이 최소인 1:1 대응. GT 가 초 단위(mm:ss)라 밀집 구간에서 앞에서부터 고르면 밀린다."""
    pairs, hit, used = [], set(), set()
    for d in gt.dir.unique():
        g, e = gt[gt.dir == d], ev[ev.dir == d]
        if g.empty or e.empty: continue
        C = np.abs(np.subtract.outer(g.time_s.values, e.time_s.values))
        for i, j in zip(*linear_sum_assignment(np.where(C <= tol, C, 1e4))):
            if C[i, j] <= tol:
                pairs.append((g.index[i], int(e.event.iloc[j]))); hit.add(g.index[i]); used.add(int(e.event.iloc[j]))
    return pairs, [i for i in gt.index if i not in hit], ev[~ev.event.isin(used)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="영상 stem (예: 260701_s1_rtspcam)")
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--pred", default=None, help="video,event,noncargo 컬럼 csv (쉼표로 여러 개)")
    ap.add_argument("--excl-work", action="store_true", help="work_cand(truck/bus 클래스)도 비화물로 처리")
    a = ap.parse_args()
    pred = None
    if a.pred:
        pred = pd.concat([pd.read_csv(p) for p in a.pred.split(",")], ignore_index=True)
        if a.excl_work and "work_cand" in pred: pred["noncargo"] = pred.noncargo | pred.work_cand
    tot = dict(gt=0, det=0, miss=0, extra=0)
    for v in a.videos:
        ev = pd.read_csv(ROOT / "runs/count" / v / "events.csv")
        gt = pd.read_csv(ROOT / "data/COUNT/gt_v2" / f"{v}.csv")
        gt["time_s"] = gt.time.map(to_sec)
        pairs, miss, extra = match(ev, gt, a.tol)
        tot["gt"] += len(gt); tot["det"] += len(ev); tot["miss"] += len(miss); tot["extra"] += len(extra)
        line = f"{v}: GT={len(gt)} events={len(ev)} 미탐={len(miss)} 오검={len(extra)}"
        if miss: line += " 미탐:" + ",".join(f"{gt.time_s[i]}s/{gt.label[i]}" for i in miss)
        if len(extra): line += " 오검:" + ",".join(f"E{r.event}@{r.time_s}s" for r in extra.itertuples())
        print(line)
        for d in sorted(gt.dir.unique()):
            gt_c = int(((gt.dir == d) & (gt.label == "cargo")).sum())
            s = f"   [{d}] 화물 GT={gt_c}"
            if pred is not None:
                p = pred[pred.video == v].set_index("event").noncargo
                pc = int(sum(1 for e in ev[ev.dir == d].event if not p.get(e, 0)))
                # 이벤트 단위 판정 오류 (매칭된 쌍 기준)
                fp = [ (gt.time_s[g], gt.label[g]) for g, e in pairs if gt.dir[g] == d and gt.label[g] == "cargo" and p.get(e, 0)]
                fn = [ (gt.time_s[g], gt.label[g]) for g, e in pairs if gt.dir[g] == d and gt.label[g] != "cargo" and not p.get(e, 0)]
                s += f" 예측={pc}  화물→비화물 오판={fp}  비화물→화물 오판={fn}"
            print(s)
    print("total", tot)


if __name__ == "__main__":
    main()
