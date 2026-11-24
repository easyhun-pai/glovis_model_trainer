"""차량 판정 규칙 변형을 events_marks.csv 로 오프라인 재판정 (재실행 없이).

현재 규칙: 이동 규칙 후 conf>=0.5 표식(종류 무관)이 있는 크롭 >= 2 → 제외.
변형 plateX: 현재 규칙 OR (이동 규칙 후 번호판 conf>=X 인 크롭 >= 1) → 제외.
사용: python count/scripts/replay_rule.py <run_dir> [--gt gt.csv | --reviewed-only] [--xs 0.5 0.6 0.7 0.8]
  --gt 가 있으면 eval_rt.py 와 같은 매칭으로 정답 라벨을 붙여 화물→제외(오제외)·제외 회복 수를 센다.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rt import mmss  # noqa: E402


def decide(m, x=None, min_frames=2):
    any_kept = (m[["plate_kept", "beacon_kept", "number_kept"]].max(axis=1) > 0)
    nf = any_kept.groupby(m.event_no).sum()
    ex = nf >= min_frames
    if x is not None:
        ex = ex | ((m.plate_kept >= x).groupby(m.event_no).sum() >= 1)
    return ex.map({True: "exclude", False: "cargo"})


def gt_labels(ev, gt_path, reviewed_only, tol=2.0):
    g = pd.read_csv(gt_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"); g["t"] = g.time.map(mmss) + 0.5
    if reviewed_only and "reviewed" in g:
        lr = float(g[g.reviewed == "1"].t.max()); nx = g[(g.reviewed != "1") & (g.t > lr)].t
        t_end = (lr + float(nx.min())) / 2 if len(nx) else lr + 1.5  # 검수된 마지막 행과 다음(미검수) 행의 중간까지
        g = g[g.reviewed == "1"]; ev = ev[ev.cross_s <= t_end]
    if "note" in g: g = g[(g.note == "") & (g.label != "invalid")]
    g = g.reset_index(drop=True); used = set(); lab = {}
    for r in ev.itertuples():
        c = g[(g.dir == r.dir) & (~g.index.isin(used)) & ((g.t - r.cross_s).abs() <= tol)]
        if len(c):
            j = (c.t - r.cross_s).abs().idxmin(); used.add(j); lab[r.event_no] = g.label[j]
    return lab, set(ev.event_no)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--gt"); ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--xs", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    a = ap.parse_args(); run = Path(a.run)
    ev = pd.read_csv(run / "events.csv", encoding="utf-8-sig"); m = pd.read_csv(run / "events_marks.csv", encoding="utf-8-sig")
    base = decide(m)
    chk = ev.set_index("event_no").decision.reindex(base.index)
    res = dict(run=run.name, logged_events=len(base), events=len(ev), base_matches_logged=int((chk == base).sum()))
    lab, scope = gt_labels(ev, a.gt, a.reviewed_only) if a.gt else ({}, set(ev.event_no))
    for x in [None] + a.xs:
        d = decide(m, x).reindex(ev.event_no).fillna("cargo"); d = d[d.index.isin(scope)]
        r = dict(cargo=int((d == "cargo").sum()), exclude=int((d == "exclude").sum()))
        if lab:
            gl = pd.Series({e: ("cargo" if l == "cargo" else "exclude") for e, l in lab.items()}); dd = d.reindex(gl.index)
            r.update(cargo_as_exclude=int(((gl == "cargo") & (dd == "exclude")).sum()), exclude_as_cargo=int(((gl == "exclude") & (dd == "cargo")).sum()),
                     work_kept_as_cargo=[f"{e}" for e, l in lab.items() if l == "work" and dd.get(e) == "cargo"],
                     wrong_events=[f"#{e} {ev.set_index('event_no').cross_time[e]} gt={lab[e]} pred={dd[e]}" for e in gl.index if (gl[e] != dd[e])])
        res["current" if x is None else f"plate>={x}"] = r
    json.dump(res, open(run / "replay_rule.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
