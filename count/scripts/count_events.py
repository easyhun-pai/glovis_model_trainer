"""COUNT 3단계: tracks.csv → 라인 통과 이벤트(카운트) 추출.

카운트 정의: 차량 bottom-center 가 영상별 통과 기준선(data/COUNT/lines.json)을 넘으면 1대.
방향: 야드→배 = 선적(load, dir=R2L), 배→야드 = 하역(unload, dir=L2R).

트랙 조각 연결(chaining):
  우측 진입부에서 차량이 프레임 안으로 들어오며 박스가 급변해 ByteTrack ID 가 끊기는 일이 잦다
  (260701_s4 에서 끊긴 구간이 라인을 건너뛰어 미탐 발생). 그래서
  트랙 A 가 끝나고 --max-gap 초 안에 시작한 트랙 B 가
    - A 의 등속 예측 위치 근처에서 시작하고
    - 진행 방향이 A 와 같으면 (속도 내적 > 0)
  같은 차량으로 보고 이어 붙인 뒤, 이어진 궤적에서 통과를 판정한다. 조각들이 각자 통과해도 1회로 합쳐진다.

기타:
- 궤적 전체 x 이동량이 --min-travel 미만이면(화면 끝 박스 흔들림 등) 통과로 치지 않는다.
- 같은 체인의 재통과(왕복)는 --min-gap 초 안이면 무시.

출력: runs/count/<영상명>/events.csv  (event, tid, chain, frame, dir, time_s[실제 재생 시각], op)
  tid = 체인에서 검출 수가 가장 많은 트랙 (크롭/분류 대표), chain = 체인 tid 목록 ('|' 구분)
"""
import argparse, json, subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def frame_pts(video, cache):
    """프레임별 실제 재생 시각(초). 평균 fps 로 나누면 영상 끊김 때문에 플레이어 시각과 어긋난다 (s4 최대 6초)."""
    cache = Path(cache)
    if cache.exists(): return np.load(cache)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(ROOT / video)], capture_output=True, text=True, check=True).stdout
    pts = np.sort([float(x.strip(",")) for x in out.split() if x.strip(",")])
    np.save(cache, pts)
    return pts


def split_id_switch(df, thr):
    """ByteTrack 이 마주 오는 두 차량에 같은 ID 를 준 경우(s2 tid3243: 화물차→세이프티카) x 진행 방향이 thr px 이상
    뒤집히는 지점에서 트랙을 나눈다. 나뉜 조각의 tid = 원래 tid + 1000000 * 조각번호."""
    df = df.sort_values(["tid", "frame"]).copy(); bx = ((df.x1 + df.x2) / 2).values; tid = df.tid.values.copy()
    for t, idx in df.groupby("tid", sort=False).indices.items():
        x = bx[idx]; piv = []; d = 0; ext = 0
        for i in range(1, len(x)):
            if d == 0:
                if abs(x[i] - x[ext]) >= thr: d = int(np.sign(x[i] - x[ext])); ext = i
                elif abs(x[i] - x[0]) < abs(x[ext] - x[0]): pass
            elif (x[i] - x[ext]) * d > 0: ext = i
            elif abs(x[i] - x[ext]) >= thr: piv.append(ext + 1); d = -d; ext = i
        for k, c in enumerate(piv, 1): tid[idx[c:]] = t + 1000000 * k
    df["tid"] = tid
    return df


def track_table(df, min_h):
    df = df.assign(bx=(df.x1 + df.x2) / 2, by=df.y2, h=df.y2 - df.y1)
    df = df[df.h >= min_h].sort_values(["tid", "frame"])
    T = {}
    for tid, g in df.groupby("tid"):
        if len(g) < 2: continue
        f = g.frame.values; p = g[["bx", "by"]].values.astype(float)
        k = min(5, len(g) - 1)
        v_end = (p[-1] - p[-1 - k]) / max(1, f[-1] - f[-1 - k])
        v_start = (p[k] - p[0]) / max(1, f[k] - f[0])
        T[tid] = dict(f=f, p=p, x1=g.x1.values.astype(float), f0=f[0], f1=f[-1], v_end=v_end, v_start=v_start)
    return T


def build_chains(T, max_gap_f, base_r, occl=None):
    order = sorted(T, key=lambda t: T[t]["f0"])
    succ, pred = {}, {}
    for b in order:
        B = T[b]; best = None
        for a, A in T.items():
            if a == b or a in succ: continue
            gap = B["f0"] - A["f1"]
            if not (0 < gap <= max_gap_f): continue
            va = A["v_end"]; sp = np.linalg.norm(va)
            if sp < 2: continue  # 정지 트랙(주차 차량)은 연결 대상 아님
            if np.dot(va, B["v_start"]) <= 0: continue  # 진행 방향 불일치
            if np.linalg.norm(B["v_start"]) < 2: continue  # 멈춰 있는 조각(램프 벽면 오검출 등)에는 잇지 않음
            pred_pos = A["p"][-1] + va * gap
            d = np.linalg.norm(B["p"][0] - pred_pos)
            if d <= base_r + 0.5 * sp * gap and (best is None or d < best[0]):
                best = (d, a)
        if best:
            succ[best[1]] = b; pred[b] = best[1]
    if occl is not None:
        # 기준선 바로 앞에서 다른 차에 가려지면 박스가 찌그러져 v_end 예측이 틀어진다 (260818_s2 27:05).
        # 선 한쪽에서 끝난 조각과 occl_gap 안에 반대쪽에서 시작한 조각이 같은 x 방향으로 가면 같은 차로 잇는다.
        side, gap_f, r = occl
        for b in order:
            if b in pred: continue
            B = T[b]; best = None
            for a, A in T.items():
                if a == b or a in succ: continue
                gap = B["f0"] - A["f1"]
                if not (0 < gap <= gap_f): continue
                if side(A["p"][-1]) == side(B["p"][0]): continue
                dxa = A["p"][-1][0] - A["p"][0][0]; dxb = B["p"][-1][0] - B["p"][0][0]
                if min(abs(dxa), abs(dxb)) < 150: continue  # 정지 물체(구석 고정 박스 등)와 잇지 않기
                if dxa * dxb <= 0 or (B["p"][0][0] - A["p"][-1][0]) * dxa <= 0: continue  # 같은 방향, B 가 A 의 앞쪽
                d = np.linalg.norm(B["p"][0] - A["p"][-1])
                if d <= r and (best is None or d < best[0]): best = (d, a)
            if best:
                succ[best[1]] = b; pred[b] = best[1]
    chains = []
    for t in order:
        if t in pred: continue
        c = [t]
        while c[-1] in succ: c.append(succ[c[-1]])
        chains.append(c)
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--u-range", type=float, nargs=2, default=[-0.5, 1.1], help="기준선 위 통과 지점 허용 범위 (0=왼끝, 1=오른끝)")
    ap.add_argument("--min-h", type=float, default=150)
    ap.add_argument("--split-x", type=float, default=250, help="트랙 x 진행 방향이 이만큼 뒤집히면 ID 스위치로 분리(px)")
    ap.add_argument("--edge-x", type=float, default=40, help="왼쪽 화면 경계에 닿았다고 보는 x1 (px). 진입 첫 박스는 x1 16~35 에서 잡힌다")
    ap.add_argument("--min-gap", type=float, default=3.0, help="같은 체인 재통과 무시(초)")
    ap.add_argument("--max-gap", type=float, default=1.5, help="조각 연결 최대 시간 간격(초)")
    ap.add_argument("--link-r", type=float, default=300, help="조각 연결 기본 거리 허용(px)")
    ap.add_argument("--min-travel", type=float, default=300, help="체인 x 이동량 최소(px)")
    ap.add_argument("--dup-gap", type=float, default=0.6, help="같은 방향 이벤트 병합 간격(초)")
    ap.add_argument("--occl-gap", type=float, default=1.0, help="기준선 양쪽에서 끊긴 조각 연결 최대 간격(초), 0=끔")
    ap.add_argument("--occl-r", type=float, default=800, help="기준선 양쪽 조각 연결 최대 거리(px)")
    ap.add_argument("--no-chain", action="store_true")
    a = ap.parse_args()
    run = Path(a.run); meta = json.load(open(run / "meta.json")); fps = meta["fps"]
    tr = split_id_switch(pd.read_csv(run / "tracks.csv"), a.split_x)
    tr.to_csv(run / "tracks_split.csv", index=False)  # classify_events 가 같은 tid 체계를 쓰도록 저장
    T = track_table(tr, a.min_h)
    lines = json.load(open(ROOT / "data/COUNT/lines.json"))
    if run.name not in lines: raise SystemExit(f"{run.name}: data/COUNT/lines.json 에 통과 기준선이 없습니다")
    A, B = np.array(lines[run.name], float); AB = B - A
    side = lambda q: AB[0] * (q[1] - A[1]) - AB[1] * (q[0] - A[0]) >= 0
    occl = (side, a.occl_gap * fps, a.occl_r) if a.occl_gap > 0 else None
    chains = [[t] for t in T] if a.no_chain else build_chains(T, a.max_gap * fps, a.link_r, occl)
    ev = []
    for c in chains:
        f = np.concatenate([T[t]["f"] for t in c]); p = np.concatenate([T[t]["p"] for t in c])
        if np.ptp(p[:, 0]) < a.min_travel: continue
        # 기준선 위쪽(y 작은 쪽) = 배, 아래쪽 = 야드. s: 야드 +1 / 배 -1
        cr = AB[0] * (p[:, 1] - A[1]) - AB[1] * (p[:, 0] - A[0]); s = np.where(cr >= 0, 1, -1); last = None; cross = []
        for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
            r = cr[i] / (cr[i] - cr[i + 1]); pm = p[i] + r * (p[i + 1] - p[i]); fr = int(round(f[i] + r * (f[i + 1] - f[i])))
            u = np.dot(pm - A, AB) / np.dot(AB, AB)
            if not (a.u_range[0] <= u <= a.u_range[1]): continue
            if last is not None and (fr - last) / fps < a.min_gap: continue
            last = fr; cross.append((fr, "R2L" if s[i] > 0 else "L2R"))
        # 선 근처에서 머뭇거리면 여러 번 넘은 것처럼 보인다 → 체인당 순방향 1회(마지막 통과)만 인정
        rep = max(c, key=lambda t: len(T[t]["f"])); name = "|".join(map(str, c)); net = "R2L" if s[0] > 0 else "L2R"
        hit = [x[0] for x in cross if x[1] == net] if s[0] != s[-1] else []
        if hit:
            ev.append(dict(tid=rep, chain=name, frame=hit[-1], dir=net))
        elif A[0] < 0:
            # 기준선 왼끝이 화면 밖인 카메라(260701): 카메라 가까운 차는 선을 넘기 전에 화면 왼쪽으로 잘려 나간다.
            # 야드 쪽에서 왼쪽 경계로 나가면 선적, 왼쪽 경계에서 나타나 야드 쪽으로 가면 하역으로 본다.
            x1 = np.concatenate([T[t]["x1"] for t in c])
            if x1[-1] <= a.edge_x and s[-1] > 0 and p[-1, 0] < p[0, 0]:
                ev.append(dict(tid=rep, chain=name, frame=int(f[-1]), dir="R2L"))
            elif x1[0] <= a.edge_x and s[0] > 0 and p[-1, 0] > p[0, 0]:
                ev.append(dict(tid=rep, chain=name, frame=int(f[0]), dir="L2R"))
    ev = pd.DataFrame(ev, columns=["tid", "chain", "frame", "dir"]).sort_values("frame").reset_index(drop=True)
    # 한 차량에 박스가 2개 잡히는 경우(트럭 적재함/차체 등) → 같은 방향 --dup-gap 초 이내 이벤트 병합.
    # 차 길이상 실제 두 대가 0.6초 안에 연속 통과하는 건 불가능 (s3 최밀집 구간도 1.4초+).
    keep = []
    for r in ev.itertuples(index=False):
        if keep and keep[-1]["dir"] == r.dir and (r.frame - keep[-1]["frame"]) / fps < a.dup_gap:
            keep[-1]["chain"] += "|" + r.chain; continue
        keep.append(r._asdict())
    ev = pd.DataFrame(keep, columns=["tid", "chain", "frame", "dir"])
    ev.insert(0, "event", range(len(ev)))
    pts = frame_pts(meta["video"], run / "pts.npy")
    ev["time_s"] = pts[ev.frame.clip(upper=len(pts) - 1).astype(int)].round(1)
    ev["op"] = ev.dir.map({"R2L": "load", "L2R": "unload"})  # 야드→배 = 선적, 배→야드 = 하역
    ev.to_csv(run / "events.csv", index=False)
    nl = sum(len(c) > 1 for c in chains)
    print(f"{run.name}: events={len(ev)} (linked chains={nl})  " + "  ".join(f"{k}={v}" for k, v in ev.dir.value_counts().items()))


if __name__ == "__main__":
    main()
