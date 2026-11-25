"""kalman_gap 단위 테스트 (GPU 불필요).  python count/scripts/test_kalman_gap.py [--core]

합성 검출을 ByteTrack 에 직접 넣어 확인한다. 속도는 실제 트랙(260818_s1) 기준: 프레임당 박스 폭의 p50 0.09, p95 0.22, 최대 0.58.
  T1 gap=0 이면 보정 설치 전과 결과가 완전히 같다 (회귀 없음)
  T2 프레임을 건너뛰어도 한 대는 끝까지 같은 id (보정 전에는 빠른 차에서 끊기는 것도 확인)
  T3 건너뛴 뒤 칼만 예측 위치가 실제 위치와 맞는다
  T4 frame_id 가 실제 경과 프레임과 같다 → lost 보관 시간이 실제 시간 기준
  T5 마주 오며 겹치는 두 차가 프레임을 건너뛰어도 id 가 뒤바뀌지 않는다
"""
import sys
from pathlib import Path

import numpy as np
from ultralytics.cfg import get_cfg  # noqa: F401
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--core" in sys.argv:  # 배포 패키지(count/deploy/app/core.py)에 복사된 구현을 검사
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "app"))
    import core as kalman_gap  # noqa: E402
else:
    import kalman_gap  # noqa: E402

CFG = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
W_IMG, H_IMG = 2560, 1440


def boxes(rows):
    a = np.array(rows, np.float32).reshape(-1, 6) if rows else np.zeros((0, 6), np.float32)
    return Boxes(a, (H_IMG, W_IMG))


def car(x, y=700, w=400, h=300, conf=0.9):
    return [x, y, x + w, y + h, conf, 2]


def run(seq, keep, comp):
    """seq: 프레임별 검출 목록. keep: 처리할 프레임 번호(오름차순). 반환: {frame: [(id, x1)]}, tracker"""
    BYTETracker.reset_id()
    tr = BYTETracker(CFG, frame_rate=30)
    if comp: kalman_gap.install(tr)
    out, prev = {}, None
    for f in keep:
        if comp and prev is not None: kalman_gap.set_gap(tr, f - prev - 1)
        r = tr.update(boxes(seq[f]))
        out[f] = [(int(t[4]), float(t[0])) for t in r]  # result: xyxy, id, score, cls, idx
        prev = f
    return out, tr


def ids(out):
    return sorted({i for v in out.values() for i, _ in v})


def t1_gap0_identical():
    rng = np.random.default_rng(0)
    seq = []
    for f in range(120):  # 차 3대 + 검출 흔들림
        seq.append([car(100 + 30 * f + rng.normal(0, 3), 650), car(2200 - 45 * f + rng.normal(0, 3), 760, 450, 330),
                    car(800 + 12 * f, 600, 300, 220, 0.6)])
    a, _ = run(seq, range(120), comp=False)
    b, _ = run(seq, range(120), comp=True)
    assert a == b, "gap=0 인데 보정 설치 후 결과가 달라짐"
    return "같음"


def t2_single_car(speed, gap, comp, n=70, w=400, start=10, warm=8):
    """빈 화면으로 시작 → start 프레임에 차 등장 → warm 프레임 정상 추적(속도 학습) → 이후 gap 장씩 건너뜀.
    실제 상황: 평소엔 다 처리하다가 순간적으로 밀려 프레임을 버림."""
    seq = [[car(50 + speed * w * (f - start), w=w)] if f >= start else [] for f in range(n)]
    keep = list(range(0, start + warm)) + list(range(start + warm, n, gap + 1))
    out, _ = run(seq, keep, comp)
    shown = [f for f in keep if f >= start + warm]
    return len(ids(out)), sum(1 for f in shown if not out[f])


def t2b_init_under_drop(speed, gap, comp, n=60, w=400, start=10):
    """처음 보이는 순간부터 프레임을 건너뛸 때 (확정 전 트랙은 예측 없이 IoU 0.3 로만 확정 — ByteTrack 구조상 한계, 기록용)."""
    seq = [[car(50 + speed * w * (f - start), w=w)] if f >= start else [] for f in range(n)]
    keep = list(range(0, start)) + list(range(start, n, gap + 1))
    out, _ = run(seq, keep, comp)
    return sum(1 for f in keep if f >= start and out[f]) > 0


def t3_prediction_error(speed=0.3, gap=3, w=400):
    seq = [[car(50 + speed * w * f, w=w)] for f in range(40)]
    keep = list(range(0, 20)) + [20 + gap + 1]  # 20프레임 정상 추적 후 gap 장 건너뜀
    res = {}
    for comp in (False, True):
        BYTETracker.reset_id(); tr = BYTETracker(CFG, frame_rate=30)
        if comp: kalman_gap.install(tr)
        prev = None
        for f in keep[:-1]:
            if comp and prev is not None: kalman_gap.set_gap(tr, f - prev - 1)
            tr.update(boxes(seq[f])); prev = f
        f = keep[-1]
        if comp: kalman_gap.set_gap(tr, f - prev - 1)
        pool = tr.joint_stracks([t for t in tr.tracked_stracks if t.is_activated], tr.lost_stracks)
        tr.multi_predict(pool)  # update 와 같은 경로로 예측만
        pred_cx = pool[0].mean[0]; true_cx = 50 + speed * w * f + w / 2
        res[comp] = abs(pred_cx - true_cx) / w
    return res


def t4_frame_id(gap=2, n=31):
    seq = [[car(50 + 20 * f)] for f in range(n)]
    keep = list(range(0, n, gap + 1))
    _, tr = run(seq, keep, comp=True)
    return tr.frame_id, keep[-1] + 1  # frame_id 는 1부터 → 마지막 처리 프레임 번호 + 1


def t5_crossing(speed, gap, comp, n=80, w=420):
    # A: 왼→오, B: 오→왼, 같은 높이에서 서로 겹치며 지나감 (B 가 약간 아래 = 카메라에 가까움)
    seq = []
    for f in range(n):
        xa = 100 + speed * w * f; xb = 100 + speed * w * (n - 1) - speed * w * f
        seq.append([car(xa, 700, w, 300), car(xb, 740, w + 30, 320)])
    keep = list(range(0, n, gap + 1))
    out, _ = run(seq, keep, comp)
    # 각 id 의 x 가 한 방향으로만 움직여야 함 (뒤바뀌면 방향이 반대로 튐)
    traj = {}
    for f in keep:
        for i, x in out[f]: traj.setdefault(i, []).append(x)
    flips = sum(1 for xs in traj.values() if len(xs) > 2 and np.any(np.diff(xs) > 0) and np.any(np.diff(xs) < 0)
                and min(np.sum(np.diff(xs) > 0), np.sum(np.diff(xs) < 0)) >= 1 and abs(xs[-1] - xs[0]) < 0.5 * np.ptp(xs))
    return len(traj), flips


def t6_lost_expiry(gap=2, comp=True):
    """차가 사라진 뒤 lost 트랙이 정리되는 실제 프레임 (ByteTrack 보관 30 = 실제 30프레임이어야 함)."""
    seq = [[car(50 + 30 * f)] if f < 20 else [] for f in range(200)]
    keep = list(range(0, 20)) + list(range(20, 200, gap + 1))
    BYTETracker.reset_id(); tr = BYTETracker(CFG, frame_rate=30)
    if comp: kalman_gap.install(tr)
    prev = None
    for f in keep:
        if comp and prev is not None: kalman_gap.set_gap(tr, f - prev - 1)
        tr.update(boxes(seq[f])); prev = f
        if f >= 20 and not tr.lost_stracks and not tr.tracked_stracks: return f - 19  # 마지막 검출(19) 뒤 몇 프레임
    return None


def t7_max_gap():
    BYTETracker.reset_id(); tr = BYTETracker(CFG, frame_rate=30); kalman_gap.install(tr)
    tr.update(boxes([car(100)])); f0 = tr.frame_id
    kalman_gap.set_gap(tr, 10**6); g = tr._gap; tr.update(boxes([]))
    return g, tr.frame_id - f0


if __name__ == "__main__":
    ok = True
    print("T1 gap=0 결과 동일:", t1_gap0_identical())
    print("\nT2 추적 중 프레임을 건너뛸 때 — (id 개수 1이 정상, 건너뛴 뒤 추적 결과에서 빠진 프레임 수 0이 정상)")
    print(f"{'속도(폭/프레임)':>14}{'gap':>5}{'보정 전':>12}{'보정 후':>12}")
    for sp in (0.09, 0.22, 0.4, 0.58):
        for g in (0, 1, 2, 3):
            a = t2_single_car(sp, g, False); b = t2_single_car(sp, g, True)
            print(f"{sp:>14}{g:>5}{str(a):>12}{str(b):>12}")
            if a[0] == 0 and b[0] == 0: print("   ^ 확정 자체가 안 됨 (보정과 무관한 ByteTrack 확정 단계 한계, T2b 참고)")
            elif b != (1, 0): ok = False; print("   ^ 보정 후에도 끊김")
    print("\nT2b (기록용) 처음부터 건너뛸 때 확정되는지 — 속도/gap: 보정 전, 보정 후")
    for sp in (0.09, 0.22, 0.4):
        print("   " + "  ".join(f"{sp}/{g}: {t2b_init_under_drop(sp, g, False)},{t2b_init_under_drop(sp, g, True)}" for g in (0, 1, 2, 3)))
    e = t3_prediction_error()
    print(f"\nT3 3프레임 건너뛴 뒤 예측 위치 오차(박스 폭 대비): 보정 전 {e[False]:.2f}, 보정 후 {e[True]:.3f}")
    if e[True] > 0.05: ok = False; print("   ^ 보정 후 오차 큼")
    fid, real = t4_frame_id()
    print(f"T4 frame_id {fid} / 실제 경과 프레임 {real}")
    if fid != real: ok = False
    print("\nT5 마주 오는 두 차 — (id 개수, 방향이 뒤집힌 id 수) 정상 (2, 0)")
    for sp in (0.09, 0.22):
        for g in (0, 1, 2):
            a = t5_crossing(sp, g, False); b = t5_crossing(sp, g, True)
            print(f"   속도 {sp} gap {g}: 보정 전 {a}  보정 후 {b}")
            if b != (2, 0): ok = False; print("   ^ 보정 후 id 개수 또는 방향 이상")
    a6, b6 = t6_lost_expiry(comp=False), t6_lost_expiry(comp=True)
    print(f"\nT6 차가 사라진 뒤 lost 트랙 정리까지 실제 프레임 (3프레임마다 처리): 보정 전 {a6}, 보정 후 {b6} (기대 약 31)")
    if b6 is None or not 29 <= b6 <= 34: ok = False; print("   ^ 보관 시간이 실제 30프레임이 아님")
    g7, df7 = t7_max_gap()
    print(f"T7 MAX_GAP 상한: 예약된 gap {g7}, frame_id 증가 {df7} (기대 {kalman_gap.MAX_GAP} / {kalman_gap.MAX_GAP + 1})")
    if (g7, df7) != (kalman_gap.MAX_GAP, kalman_gap.MAX_GAP + 1): ok = False
    print("\n결과:", "통과" if ok else "실패"); sys.exit(0 if ok else 1)
