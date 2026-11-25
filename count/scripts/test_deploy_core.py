"""배포 엔진(count/deploy/app) 회귀 테스트 — QA 리뷰(2026-11-25) 지적 사항.  python count/scripts/test_deploy_core.py [--gpu]

  D1 재접속(reset_tracks) 뒤 같은 ByteTrack id 를 받은 새 차도 센다            (리뷰 HIGH-1)
  D2 RTSP 프레임이 몰려 들어와도 프레임 번호는 1씩 (칼만 보정이 과하게 안 감)    (HIGH-2)
     재접속 때만 끊긴 시간만큼 건너뜀
  D3 모양 급변 분리는 프레임 간격과 무관 (lost 후 벽면 박스로 재활성)           (MEDIUM-4)
  D4 폭 0 박스에서 0 나누기 없음                                              (LOW-7)
  D5 자정 넘김: 상태 갱신만으로 '오늘'이 바뀌고, 통과 날짜 기준으로 센다           (LOW-8)
  D6 [--gpu] 깨진 TensorRT 엔진 → PyTorch 로 넘어가고 엔진은 .bad 로 치움          (MEDIUM-3)
"""
import datetime as dt
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "deploy" / "app"
sys.path.insert(0, str(APP))
import core  # noqa: E402
import count_live  # noqa: E402
import storage  # noqa: E402


def stub_counter():
    """모델 없이 판정 로직만 도는 Counter (리뷰어 재현 스크립트 기반)."""
    C = object.__new__(core.Counter)
    C.W, C.H, C.fps = 2560, 1440, 30.0
    C.min_h, C.split_x, C.edge_x = 150, 250, 40
    C.link_r, C.min_travel, C.occl_r = 300, 300, 800
    C.still, C.occl_min_dx = 2.0, 150
    C.max_gap, C.occl_gap, C.min_gap, C.dup_gap = 1.5, 1.0, 3.0, 0.6
    C.finalize, C.min_crops, C.grace = 0.7, 3, 0.5
    C.u_range, C.k, C.keep_crops = [-0.5, 1.1], 8, 24
    C.pad_x, C.pad_top = 0.06, 0.3
    C.LA, C.LB = np.array([1280., 0.]), np.array([1280., 1440.]); C.AB = C.LB - C.LA; C.ysign = 1
    C.on_event = C.on_decision = None

    class MW: q = queue.Queue()
    C.mw = MW(); C.rg = None
    C.raw = {}; C.consumed = set(); C.events = []; C.lock = threading.Lock()
    C.emitted = {}; C.pending = []; C.n_events = 0; C.last_fi = None; C.kalman_gap = False

    class Tr:
        def reset(self): pass

    class P: trackers = [Tr()]

    class D: predictor = P()
    C.det = D()
    return C


def drive(C, tid, f0, x0, vx, n=60):
    for i in range(n):
        f = f0 + i; x = x0 + vx * i
        rt = C.raw.get(tid) or C.raw.setdefault(tid, core.RawTrack(tid))
        if rt.pid(rt.k) in C.consumed and rt.rows: continue
        rt.add((f, 2, 0.9, x, 600, x + 500, 1000), C.split_x)
        C.evaluate(f)


def d1():
    C = stub_counter()
    drive(C, 1, 0, 300, 30); C.evaluate(10**6)
    n0 = len(C.events)
    C.reset_tracks()
    drive(C, 1, 2000, 300, 30); C.evaluate(10**7, flush=True)
    return n0, len(C.events)


def d2():
    # 지연 뒤 몰려 들어온 프레임: 도착 시각과 무관하게 +1
    seq, last, t_last = [], -1, None
    for now in [0.0, 0.033, 0.066, 0.4, 0.401, 0.402, 0.403, 0.5]:
        last = count_live.next_fi(last, False, now, t_last, 30.0); t_last = now; seq.append(last)
    burst_ok = seq == list(range(8))
    # 재접속: 3초 끊김 → 약 90 프레임 건너뜀, 다음부터 다시 +1
    j = count_live.next_fi(7, True, 3.5, 0.5, 30.0); k = count_live.next_fi(j, False, 3.53, 3.5, 30.0)
    return burst_ok, seq, j, k


def d3():
    rt = core.RawTrack(1)
    for f in range(10): rt.add((f, 2, 0.9, 300 + 40 * f, 600, 900 + 40 * f, 1050), 250)   # 차 (종횡비 1.33)
    for f in range(25, 29): rt.add((f, 2, 0.9, 270, 535, 1370, 785), 250)                 # 15프레임 lost 뒤 벽면 (4.4)
    return rt.piece


def d4():
    rt = core.RawTrack(1)
    try:
        rt.add((0, 2, 0.9, 100, 600, 100, 1000), 250); rt.add((1, 2, 0.9, 100, 600, 500, 1000), 250); return True
    except ZeroDivisionError:
        return False


def d5():
    tmp = Path(tempfile.mkdtemp())
    try:
        db = storage.EventDB(tmp / "c.db")
        m = object.__new__(storage.MediaWriter)
        m.db = db; m.cam = "cam"; m.lock = threading.Lock(); m.root = tmp; m.run_id = db.start_run(camera_id="cam")
        m.session = dict(load_cargo=0, unload_cargo=0, exclude=0); m.last = None
        m.day = dt.date.today() - dt.timedelta(days=1); m.today = dict(load_cargo=5, unload_cargo=5, exclude=5)  # 어제 누적이 남은 상태
        m.status(dict(running=True))
        rolled = m.day == dt.date.today() and m.today == dict(load_cargo=0, unload_cargo=0, exclude=0)
        # 어제 23:59:59 에 통과했는데 오늘 기록되는 차 → 오늘 누적에 넣지 않음
        yday = (dt.datetime.combine(dt.date.today(), dt.time()) - dt.timedelta(seconds=1)).timestamp()
        ev = dict(frame=0, op="load", decision="cargo", marks={}, n_crops=8, reason="", decided_wall=yday + 1, no=1)
        m.wall_of = lambda fr: yday; m.save_shot = m.save_clip = m.save_crop = False; m.log = lambda *a: None
        m.pending = []; m._csv = lambda *a: None
        m._record(ev)
        not_today = m.today["load_cargo"] == 0 and m.session["load_cargo"] == 1
        db.c.close()
        return rolled, not_today
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def d6():
    import torch
    if not torch.cuda.is_available(): return None
    import re
    import tensorrt
    tmp = Path(tempfile.mkdtemp())
    try:
        pt = tmp / "marks_r3.pt"; shutil.copy(Path(__file__).resolve().parents[1] / "models" / "marks_r3.pt", pt)
        gpu = re.sub(r"[^A-Za-z0-9]+", "", torch.cuda.get_device_name(0)); trtv = re.sub(r"[^0-9.]+", "", tensorrt.__version__.split("+")[0])
        bad = tmp / f"marks_r3_640_b8dyn_{gpu}_trt{trtv}.engine"; bad.write_bytes(b"not an engine" * 100)
        logs = []
        m, ok = core.load_model(str(pt), True, 640, batch=8, dynamic=True, log=logs.append, warm=[[np.zeros((480, 640, 3), np.uint8)]])
        return (not ok) and (not bad.exists()) and bad.with_suffix(".engine.bad").exists() and str(m.ckpt_path).endswith(".pt"), logs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ok = True
    n0, n1 = d1(); r = (n0, n1) == (1, 2); ok &= r
    print(f"D1 재접속 뒤 같은 id 의 새 차도 셈: {r}  (재접속 전 {n0}대 → 후 {n1}대, 기대 1 → 2)")
    b, seq, j, k = d2(); r = b and 95 <= j <= 97 and k == j + 1; ok &= r
    print(f"D2 몰려 들어와도 +1, 재접속 땐 끊긴 시간만큼: {r}  ({seq}, 재접속 7→{j}, 다음 {k})")
    p = d3(); r = p == [0] * 10 + [1] * 4; ok &= r
    print(f"D3 15프레임 끊긴 뒤 벽면으로 넘어가도 분리: {r}  {p}")
    r = d4(); ok &= r
    print(f"D4 폭 0 박스 안전: {r}")
    rolled, nt = d5(); ok &= rolled and nt
    print(f"D5 자정 넘김 상태 갱신으로 '오늘' 초기화: {rolled} · 어제 통과 차는 오늘 누적 제외: {nt}")
    if "--gpu" in sys.argv:
        res = d6()
        if res is None: print("D6 GPU 없음 — 건너뜀")
        else:
            r, logs = res; ok &= r
            print(f"D6 깨진 TensorRT 엔진 → PyTorch + .bad 로 치움: {r}  ({logs[-1][:80] if logs else ''})")
    print("\n결과:", "통과" if ok else "실패"); sys.exit(0 if ok else 1)
