"""트랙 조각 규칙 단위 테스트 (GPU 불필요).  python count/scripts/test_tracking_rules.py

  R1 박스 모양(가로/세로)이 1.8배 넘게 급변하면 새 조각 — 램프 벽면으로 id 가 넘어간 경우 (R1b: 프레임 간격 > 5)
  R2 정상 차량(화면 가장자리로 들어오며 커지는 박스, 프레임 건너뜀 포함)은 나뉘지 않음
  R3 x 진행 방향 반전 분리(기존 규칙)는 그대로
  R4 조각 연결: 움직이던 조각 뒤에 멈춰 있는 조각(벽면)은 잇지 않음, 움직이는 조각은 이음
시뮬레이터(realtime_sim / count_events)와 배포 엔진(core) 두 구현을 같은 입력으로 검사한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "deploy" / "app"))
import core  # noqa: E402
import count_events  # noqa: E402
import realtime_sim  # noqa: E402


def feed(RawTrack, rows, split_x=250):
    rt = RawTrack(1)
    for r in rows: rt.add(r, split_x)
    return rt.piece


def row(f, x1, y1, x2, y2): return (f, 2, 0.9, x1, y1, x2, y2)


def r1():
    # 차(600x450, 종횡비 1.33) 왼→오 이동 → 벽면 박스(1100x250, 4.4) 로 id 넘어감
    rows = [row(f, 300 + 40 * f, 600, 900 + 40 * f, 1050) for f in range(10)] + [row(f, 270, 535, 1370, 785) for f in range(10, 14)]
    return rows, [0] * 10 + [1] * 4


def r1b():
    # 15프레임 lost 됐다가 같은 id 가 벽면 박스로 재활성 (간격 > 5)
    rows = [row(f, 300 + 40 * f, 600, 900 + 40 * f, 1050) for f in range(10)] + [row(f, 270, 535, 1370, 785) for f in range(25, 29)]
    return rows, [0] * 10 + [1] * 4


def r2():
    # 오른쪽 가장자리로 들어오며 폭이 커지는 차 (종횡비 0.9→1.3 을 여러 프레임에 걸쳐) + 2프레임 건너뜀
    rows = []
    for f in list(range(0, 8)) + list(range(10, 20, 3)):
        w = min(600, 300 + 50 * f); rows.append(row(f, 2560 - w, 600, 2560, 1050))
    return rows, [0] * len(rows)


def r3():
    rows = [row(f, 300 + 60 * f, 600, 900 + 60 * f, 1050) for f in range(10)] + \
           [row(f, 840 - 60 * (f - 10), 600, 1440 - 60 * (f - 10), 1050) for f in range(10, 20)]
    return rows


def r4(build_chains, still_kw):
    def piece(tid, f0, n, x0, vx, w=600, h=450):
        return [(f0 + i, tid, 2, 0.9, x0 + vx * i, 600, x0 + vx * i + w, 600 + h) for i in range(n)]
    moving = piece(1, 0, 15, 200, 40)
    # 멈춰 있는 조각: 검출 흔들림으로 진행 방향과 같은 쪽으로 1px/프레임 (내적 > 0 이라 예전 방향 검사로는 못 거름)
    wall = [(f, 2, 2, 0.9, 270 + (f - 15), 535, 1370 + (f - 15), 1085) for f in range(15, 22)]
    cont = piece(3, 16, 10, 200 + 40 * 16, 40)                               # 같은 차가 이어진 조각
    df = pd.DataFrame(moving + wall + cont, columns=["frame", "tid", "cls", "conf", "x1", "y1", "x2", "y2"])
    T = count_events.track_table(df, 150)
    ch = build_chains(T, 1.5 * 20, 300, None, **still_kw)
    return {tuple(c) for c in ch}


if __name__ == "__main__":
    ok = True
    impls = {"시뮬레이터": realtime_sim.RawTrack, "배포 core": core.RawTrack}
    for name, RT in impls.items():
        rows, want = r1(); got = feed(RT, rows)
        print(f"R1 [{name}] 벽면으로 넘어가면 분리: {got == want}  {got}"); ok &= got == want
        rows, want = r1b(); got = feed(RT, rows)
        print(f"R1b [{name}] 15프레임 끊긴 뒤 벽면으로 넘어가도 분리: {got == want}  {got}"); ok &= got == want
        rows, want = r2(); got = feed(RT, rows)
        print(f"R2 [{name}] 가장자리 진입 차량은 안 나뉨: {got == want}  {got}"); ok &= got == want
        got = feed(RT, r3()); n = len(set(got))
        print(f"R3 [{name}] 방향 반전 분리 유지: {n == 2}  {got}"); ok &= n == 2
    want = {(1, 3), (2,)}
    for name, (bc, kw) in {"시뮬레이터(count_events)": (count_events.build_chains, {}),
                           "배포 core": (core.build_chains, {"still": 2.0, "occl_min_dx": 150})}.items():
        got = r4(bc, kw)
        print(f"R4 [{name}] 멈춘 조각에 안 잇고 움직이는 조각에 이음: {got == want}  {sorted(got)}"); ok &= got == want
    print("\n결과:", "통과" if ok else "실패"); sys.exit(0 if ok else 1)
