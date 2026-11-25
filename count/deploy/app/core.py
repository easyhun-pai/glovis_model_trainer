"""COUNT 실시간 판정 엔진 (현장 배포용, 저장소 코드에 의존하지 않음).

평가에 쓴 count/scripts/realtime_sim.py (--decide-at cross) 와 같은 로직:
  1) 차량 탐지 yolo26m + ByteTrack
  2) ID 스위치 분리 · 끊긴 조각 연결(등속 예측 / 기준선 양쪽 가림) → 체인
  3) 체인의 하단 중심이 기준선을 넘는 순간 1대 (야드→배 = 선적, 배→야드 = 하역)
  4) 접근 중 모은 차량 크롭 최대 8장 → 표식 모델(번호판·경광등·창문 번호)
     차량과 같이 움직이는 표식만 인정, conf≥0.5 표식이 2장 이상이면 제외(세이프티·작업차), 아니면 화물

픽셀 단위 파라미터는 학습 현장(2560x1440) 기준이라 해상도가 다르면 가로 폭 비율로 자동 환산한다.
"""
import queue
import threading
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

VEHICLE = [2, 5, 7]                      # COCO car, bus, truck
CLS = {0: "plate", 1: "beacon", 2: "number"}
AR_JUMP = 1.8                            # 트랙 조각 분리: 연속 검출 간 가로세로 비율 변화 상한
REF_W = 2560                             # 픽셀 파라미터 기준 해상도(가로)


# ---------------------------------------------------------------- 트랙 조각 · 체인
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


def build_chains(T, max_gap_f, base_r, occl=None, still=2.0, occl_min_dx=150):
    order = sorted(T, key=lambda t: T[t]["f0"])
    succ, pred = {}, {}
    for b in order:  # 등속 예측 위치 근처에서 같은 방향으로 이어지는 조각
        B = T[b]; best = None
        for a, A in T.items():
            if a == b or a in succ: continue
            gap = B["f0"] - A["f1"]
            if not (0 < gap <= max_gap_f): continue
            va = A["v_end"]; sp = np.linalg.norm(va)
            if sp < still: continue
            if np.dot(va, B["v_start"]) <= 0: continue  # 진행 방향 불일치
            if np.linalg.norm(B["v_start"]) < still: continue  # 멈춰 있는 조각(램프 벽면 오검출 등)에는 잇지 않음
            d = np.linalg.norm(B["p"][0] - (A["p"][-1] + va * gap))
            if d <= base_r + 0.5 * sp * gap and (best is None or d < best[0]): best = (d, a)
        if best: succ[best[1]] = b; pred[b] = best[1]
    if occl is not None:  # 기준선 바로 앞에서 가려져 끊긴 조각: 선 양쪽에서 같은 x 방향이면 잇는다
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
                if min(abs(dxa), abs(dxb)) < occl_min_dx: continue
                if dxa * dxb <= 0 or (B["p"][0][0] - A["p"][-1][0]) * dxa <= 0: continue
                d = np.linalg.norm(B["p"][0] - A["p"][-1])
                if d <= r and (best is None or d < best[0]): best = (d, a)
            if best: succ[best[1]] = b; pred[b] = best[1]
    chains = []
    for t in order:
        if t in pred: continue
        c = [t]
        while c[-1] in succ: c.append(succ[c[-1]])
        chains.append(c)
    return chains


class RawTrack:
    """ByteTrack 원본 id 하나. x 진행 방향이 split_x 이상 뒤집히면(마주 오는 두 차에 같은 id) 조각 번호를 올린다."""
    def __init__(self, tid):
        self.tid = tid; self.rows = []; self.piece = []; self.d = 0; self.ext = 0; self.k = 0
        self.crops = {}; self.stride = 1; self.nc = 0

    def add(self, row, thr):
        self.rows.append(row); self.piece.append(self.k); i = len(self.rows) - 1
        xs = lambda j: (self.rows[j][3] + self.rows[j][5]) / 2
        if i == 0: return
        # 박스 모양(가로/세로)이 1.8배 넘게 급변 → 추적 id 가 다른 물체(램프 벽면 등)로 넘어간 것 → 새 조각.
        # 프레임 간격과 무관하게 비교 (lost 됐다가 벽면 박스로 재활성되는 경우 포함).
        # 정상 트랙 연속 검출 약 9.8만 쌍(5개 영상, 간격 1~90프레임)에서 최대 1.59배 (2026-11-25 측정).
        p, q = self.rows[i - 1], self.rows[i]
        ar = lambda r: max(1.0, r[5] - r[3]) / max(1.0, r[6] - r[4])
        if max(ar(p) / ar(q), ar(q) / ar(p)) > AR_JUMP:
            self.k += 1; self.piece[i] = self.k; self.d = 0; self.ext = i
            return
        x = xs(i)
        if self.d == 0:
            if abs(x - xs(self.ext)) >= thr: self.d = int(np.sign(x - xs(self.ext))); self.ext = i
        elif (x - xs(self.ext)) * self.d > 0: self.ext = i
        elif abs(x - xs(self.ext)) >= thr:
            self.k += 1
            for j in range(self.ext + 1, i + 1): self.piece[j] = self.k
            self.d = -self.d; self.ext = i

    def pid(self, k): return self.tid + 1000000 * k


# ---------------------------------------------------------------- 칼만 예측 프레임 간격 보정 (count/scripts/kalman_gap.py 와 동일)
# ByteTrack 은 update 1회 = 1프레임으로 가정한다. 프레임을 k장 건너뛰면 칼만 예측을 1+k 번 돌리고 frame_id 도 k 만큼 올린다.
MAX_GAP = 60  # 이보다 길게 끊기면(재접속 등) 예측이 의미 없음 — 상한만 두고 그대로 진행 (트랙은 lost 처리됨)


def install(tracker):
    """tracker(BYTETracker/BOTSORT) 에 간격 보정을 한 번 설치. 이미 설치돼 있으면 그대로."""
    if getattr(tracker, "_gap_installed", False): return tracker
    orig = tracker.multi_predict
    tracker._gap = 0

    def multi_predict(tracks):
        n = 1 + tracker._gap
        tracker._gap = 0  # 이번 update 에서만 적용
        for _ in range(n): orig(tracks)

    tracker.multi_predict = multi_predict
    tracker._gap_installed = True
    return tracker


def set_gap(tracker, gap):
    """다음 update() 전에 호출. gap = 지난 처리 프레임과 이번 프레임 사이에 건너뛴 프레임 수."""
    gap = int(max(0, min(gap, MAX_GAP)))
    install(tracker)
    tracker._gap = gap
    tracker.frame_id += gap  # update() 가 +1 → 합계 gap+1 = 실제 경과 프레임


def tracker_of(model):
    """YOLO.track(persist=True) 이 만든 추적기 (첫 track 호출 전이면 None)."""
    p = getattr(model, "predictor", None)
    tr = getattr(p, "trackers", None) if p is not None else None
    return tr[0] if tr else None


# ---------------------------------------------------------------- 모델 불러오기 (TensorRT 우선)
def det_shape(W, H, imgsz=640, stride=32):
    """PyTorch 의 rect 추론과 같은 입력 크기 (긴 변 imgsz, 짧은 변은 stride 배수로 올림). 2560x1440 → (384, 640)."""
    if W >= H: return (int(np.ceil(imgsz * H / W / stride) * stride), imgsz)
    return (imgsz, int(np.ceil(imgsz * W / H / stride) * stride))


def load_model(pt, use_trt, imgsz, batch=1, dynamic=False, log=print, warm=None):
    """use_trt 이면 이 PC 의 GPU 용 TensorRT 엔진을 쓴다. 없으면 처음 한 번 만들어 models/ 에 저장 (수 분).
    TensorRT 가 없거나 실패하면 PyTorch(.pt) 로 실행."""
    import torch
    if not use_trt or not torch.cuda.is_available(): return YOLO(pt), False
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        log("TensorRT 가 설치돼 있지 않아 PyTorch 로 실행합니다"); return YOLO(pt), False
    import re
    from pathlib import Path
    import tensorrt
    gpu = re.sub(r"[^A-Za-z0-9]+", "", torch.cuda.get_device_name(0))
    trtv = re.sub(r"[^0-9.]+", "", tensorrt.__version__.split("+")[0])
    sz = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, (tuple, list)) else f"{imgsz}"
    # 엔진은 GPU·TensorRT 버전에 묶인다 → 이름에 둘 다 넣어 바뀌면 새로 만든다
    eng = Path(pt).with_name(f"{Path(pt).stem}_{sz}_b{batch}{'dyn' if dynamic else ''}_{gpu}_trt{trtv}.engine")
    try:
        if not eng.exists():
            log(f"TensorRT 엔진 만드는 중: {eng.name} (이 PC 에서 처음 한 번, 수 분 걸림)")
            out = YOLO(pt).export(format="engine", imgsz=imgsz, half=True, batch=batch, dynamic=dynamic, device=0, verbose=False)
            Path(out).replace(eng)
            Path(pt).with_suffix(".onnx").unlink(missing_ok=True)  # 중간 파일 정리
        m = YOLO(str(eng), task="detect")
        for x in (warm or []):  # 엔진은 첫 추론 때 읽힌다 → 여기서 실제로 돌려 봐야 깨진 엔진을 잡는다
            m.predict(x, imgsz=imgsz, device=0, half=True, verbose=False)
        return m, True
    except Exception as e:
        log(f"TensorRT 엔진 실패 → PyTorch 로 실행: {e!r}")
        if eng.exists():  # 다음 실행 때 다시 고르지 않도록 치워 둔다 (다음 실행에서 새로 만듦)
            try: eng.replace(eng.with_suffix(".engine.bad"))
            except OSError: pass
        return YOLO(pt), False


# ---------------------------------------------------------------- 표식 판정
def motion_static(D):
    """차량이 실제로 움직인 크롭 쌍에서 화면 좌표가 고정(<0.04w)이고 차량 기준 같은 자리(<0.10)로 따라 움직인 짝이 없으면
    고정 구조물(과속 경고등·볼라드 불빛·도색 글자)로 보고 버린다. D: (crop_i, cls, conf, gx, gy, bx1, bx2, vw, rx, ry)"""
    static = set()
    for k in CLS:
        idx = [n for n, d in enumerate(D) if d[1] == k]
        for i in idx:
            a = D[i]; fixed = follow = False
            for j in idx:
                b = D[j]
                if a[0] == b[0]: continue
                d1, d2 = b[5] - a[5], b[6] - a[6]
                if not (d1 * d2 > 0 and min(abs(d1), abs(d2)) > 0.25 * a[7]): continue
                if np.hypot(a[3] - b[3], a[4] - b[4]) < 0.04 * a[7]: fixed = True
                if abs(a[8] - b[8]) < 0.10 and abs(a[9] - b[9]) < 0.10: follow = True
            if fixed and not follow: static.add(i)
    return static


class MarkWorker(threading.Thread):
    """표식 판정 전용 스레드 (차량 탐지 루프를 막지 않음)."""
    def __init__(self, model, conf, min_frames, device, half):
        super().__init__(daemon=True)
        self.m = model; self.conf = conf; self.min_frames = min_frames; self.device = device; self.half = half
        self.q = queue.Queue(); self.lat = []
        for n in (1, 8): self.m.predict([np.zeros((480, 640, 3), np.uint8)] * n, imgsz=640, device=device, half=half, verbose=False)

    def run(self):
        while True:
            job = self.q.get()
            if job is None: break
            ev, crops, cb = job; t = time.perf_counter()
            try:
                res = self.m.predict([c["img"] for c in crops], imgsz=640, conf=0.25, device=self.device, half=self.half, verbose=False) if crops else []
                D = []
                for i, (c, r) in enumerate(zip(crops, res)):
                    bw = c["bx2"] - c["bx1"]; bh = c["by2"] - c["by1"]
                    for (x1, y1, x2, y2), cf, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
                        if cf < self.conf: continue
                        gx = c["cx1"] + (x1 + x2) / 2; gy = c["cy1"] + (y1 + y2) / 2
                        D.append((i, k, cf, gx, gy, c["bx1"], c["bx2"], bw, (gx - c["bx1"]) / bw, (gy - c["by1"]) / bh))
                static = motion_static(D)
                hits = {}
                for n, d in enumerate(D):
                    if n not in static: hits.setdefault(d[0], set()).add(CLS[d[1]])
                cnt = {k: sum(k in s for s in hits.values()) for k in CLS.values()}
                excl = len(hits) >= self.min_frames
                reason = f"{len(hits)}/{len(crops)} 크롭 " + " ".join(f"{k}{v}" for k, v in cnt.items() if v)
                dt = time.perf_counter() - t; self.lat.append(dt)
                cb(ev, "exclude" if excl else "cargo", reason.strip(), cnt, dt)
            except Exception as e:  # 판정 실패는 화물로 두고 사유를 남긴다 (카운트 누락보다 나음)
                cb(ev, "cargo", f"표식 판정 오류: {e!r}", {}, time.perf_counter() - t)


# ---------------------------------------------------------------- 스트림 재전송 감지
class ReplayGuard:
    """RTSP 재접속·버퍼 재전송으로 몇 초 전 영상이 다시 들어오면 그 구간을 추적·카운트에서 뺀다 (재인코딩 수준으로 거의 같은 프레임만)."""
    def __init__(self, fps, buf_s=5.0, min_age_s=0.5, motion=1.0, rel=0.2, absd=0.6, k=3):
        self.n = max(2, int(buf_s * fps)); self.min_age = max(1, int(min_age_s * fps)); self.motion = motion; self.rel = rel; self.abs = absd; self.k = k
        self.B = np.zeros((self.n, 90, 160), np.float32); self.cnt = 0; self.prev = None; self.run = 0; self.miss = 0; self.on = False; self.segments = 0

    def check(self, img):
        g = cv2.cvtColor(cv2.resize(img[::4, ::4], (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mot = float(np.abs(g - self.prev).mean()) if self.prev is not None else 0.0
        match = False; m = min(self.cnt, self.n)
        if (mot > self.motion or self.on) and m > self.min_age:
            ages = np.arange(self.min_age, m + 1); idx = (self.cnt - ages) % self.n
            best = float(np.abs(self.B[idx] - g).mean((1, 2)).min())
            match = best < (self.abs if self.on else min(self.abs, self.rel * mot))
        if match:
            self.run += 1; self.miss = 0
            if not self.on and self.run >= self.k: self.on = True; self.segments += 1
        else:
            self.run = 0
            if self.on:
                self.miss += 1
                if self.miss >= 2: self.on = False
        self.prev = g; self.B[self.cnt % self.n] = g; self.cnt += 1
        return self.on


# ---------------------------------------------------------------- 카운터
class Counter:
    """process(fi, img) 를 프레임마다 부른다. fi 는 카메라에서 받은 프레임 번호(버린 프레임 포함, 시간축).
    이벤트가 생기면 on_event(ev) (선 통과 순간, decision=None), 판정이 끝나면 on_decision(ev) 를 부른다."""

    def __init__(self, cfg, W, H, fps, line, yard_side, on_event=None, on_decision=None):
        import torch
        p = cfg["params"]; s = W / REF_W
        self.W, self.H, self.fps = W, H, fps
        self.device = 0 if torch.cuda.is_available() else "cpu"; self.half = self.device == 0
        px = lambda k: p[k] * s
        self.min_h, self.split_x, self.edge_x = px("min_h"), px("split_x"), px("edge_x")
        self.link_r, self.min_travel, self.occl_r = px("link_r"), px("min_travel"), px("occl_r")
        self.still, self.occl_min_dx = 2.0 * s, 150 * s
        self.max_gap, self.occl_gap, self.min_gap, self.dup_gap = p["max_gap_s"], p["occl_gap_s"], p["min_gap_s"], p["dup_gap_s"]
        self.finalize, self.min_crops, self.grace = p["finalize_s"], p["min_crops"], p["cross_grace_s"]
        self.u_range, self.k, self.keep_crops = p["u_range"], p["crops_per_vehicle"], p["keep_crops"]
        self.pad_x, self.pad_top = p["pad_x"], p["pad_top"]
        self.det_conf, self.imgsz = p["det_conf"], p["imgsz"]
        self.LA, self.LB = np.array(line[0], float), np.array(line[1], float); self.AB = self.LB - self.LA
        self.ysign = 1 if yard_side >= 0 else -1  # 야드 쪽 외적 부호
        self.on_event, self.on_decision = on_event, on_decision

        trt = cfg.get("use_tensorrt", True) and self.device == 0; log = cfg.get("_log", print)
        self.imgsz = det_shape(W, H, self.imgsz)  # 직사각 입력 (PyTorch rect 추론과 같은 크기)
        self.det, self.det_trt = load_model(cfg["models"]["vehicle"], trt, self.imgsz, log=log, warm=[np.zeros((H, W, 3), np.uint8)])
        self.det.predict(np.zeros((H, W, 3), np.uint8), imgsz=self.imgsz, device=self.device, half=self.half, verbose=False)
        crop0 = np.zeros((480, 640, 3), np.uint8)
        mm, self.mark_trt = load_model(cfg["models"]["marks"], trt, 640, batch=8, dynamic=True, log=log, warm=[[crop0], [crop0] * 8])
        if self.mark_trt and self.k > 8:  # 표식 엔진은 한 번에 최대 8장
            log(f"crops_per_vehicle {self.k} → 8 (TensorRT 표식 엔진 최대 배치)"); self.k = 8
        self.mw = MarkWorker(mm, p["mark_conf"], p["mark_min_crops"], self.device, self.half); self.mw.start()
        self.rg = ReplayGuard(fps) if cfg.get("replay_guard", True) else None

        self.raw = {}; self.consumed = set(); self.events = []; self.lock = threading.Lock()
        self.emitted = {}; self.pending = []; self.n_events = 0; self.last_fi = None; self.kalman_gap = cfg.get("kalman_gap", True); self.det_lat = []; self.proc = 0; self.replay_skipped = 0

    # 야드 쪽 +1, 배 쪽 -1
    def cross_val(self, q): return self.ysign * (self.AB[0] * (q[..., 1] - self.LA[1]) - self.AB[1] * (q[..., 0] - self.LA[0]))
    def side(self, q): return self.cross_val(np.asarray(q, float)) >= 0

    def process(self, fi, img):
        """반환: 화면 표시용 [(tid, box, event or None)]"""
        if self.rg is not None and self.rg.check(img):
            self.replay_skipped += 1; return []
        tr = tracker_of(self.det)
        if tr is not None and self.last_fi is not None and self.kalman_gap: set_gap(tr, fi - self.last_fi - 1)
        t = time.perf_counter()
        r = self.det.track(img, persist=True, imgsz=self.imgsz, conf=self.det_conf, classes=VEHICLE, tracker="bytetrack.yaml",
                           device=self.device, half=self.half, verbose=False)[0]
        self.det_lat.append(time.perf_counter() - t); self.det_lat = self.det_lat[-2000:]; self.proc += 1; self.last_fi = fi
        vis = []
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cl, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                rt = self.raw.get(tid)
                if rt is None: rt = self.raw[tid] = RawTrack(tid)
                if rt.pid(rt.k) in self.consumed and rt.rows:
                    continue
                rt.add((fi, cl, cf, x1, y1, x2, y2), self.split_x)
                e = self.emitted.get(rt.pid(rt.k))
                b = e.get("boxes") if e is not None else None
                if b is not None: b[fi] = (x1, y1, x2, y2)  # 통과 후 궤적 (클립 표시용)
                if y2 - y1 >= self.min_h:
                    rt.nc += 1
                    if (rt.nc - 1) % rt.stride == 0:
                        w, h = x2 - x1, y2 - y1
                        cx1, cy1 = max(0, int(x1 - self.pad_x * w)), max(0, int(y1 - self.pad_top * h))
                        cx2, cy2 = min(self.W, int(x2 + self.pad_x * w)), min(self.H, int(y2))
                        rt.crops[len(rt.rows) - 1] = dict(frame=fi, img=img[cy1:cy2, cx1:cx2].copy(), cx1=cx1, cy1=cy1, bx1=x1, by1=y1, bx2=x2, by2=y2)
                        if len(rt.crops) > self.keep_crops:
                            rt.stride *= 2
                            for j in sorted(rt.crops)[1::2]: del rt.crops[j]
                vis.append((tid, (x1, y1, x2, y2), self.emitted.get(rt.pid(rt.k))))
        self.evaluate(fi)
        return vis

    def pieces_df(self):
        rows = []
        for rt in self.raw.values():
            n = len(rt.rows)
            sel = range(n) if n <= 300 else [*range(10), *np.linspace(10, n - 151, 100).astype(int).tolist(), *range(n - 150, n)]
            for q in sel:
                r, k = rt.rows[q], rt.piece[q]; pid = rt.pid(k)
                if pid in self.consumed: continue
                rows.append((r[0], pid, r[1], r[2], r[3], r[4], r[5], r[6]))
        return pd.DataFrame(rows, columns=["frame", "tid", "cls", "conf", "x1", "y1", "x2", "y2"])

    def chain_boxes(self, chain):
        """체인의 프레임별 차량 박스 {fi: (x1,y1,x2,y2)} — 스크린샷·클립에서 이 차를 표시하는 데 쓴다."""
        out = {}
        for pid in chain:
            rt = self.raw.get(pid % 1000000)
            if rt is None: continue
            k = pid // 1000000
            for r, pk in zip(rt.rows, rt.piece):
                if pk == k: out[r[0]] = (r[3], r[4], r[5], r[6])
        return out

    def crops_for(self, chain):
        cs = []
        for pid in chain:
            rt = self.raw.get(pid % 1000000)
            if rt is None: continue
            k = pid // 1000000
            cs += [c for j, c in rt.crops.items() if rt.piece[j] == k]
        cs.sort(key=lambda c: c["frame"])
        if len(cs) > self.k: cs = [cs[i] for i in np.unique(np.linspace(0, len(cs) - 1, self.k).round().astype(int))]
        return cs

    def crossing(self, c, T, allow_edge):
        f = np.concatenate([T[t]["f"] for t in c]); p = np.concatenate([T[t]["p"] for t in c])
        o = np.argsort(f, kind="stable"); f, p = f[o], p[o]
        if np.ptp(p[:, 0]) < self.min_travel: return None
        cr = self.cross_val(p); s = np.where(cr >= 0, 1, -1); last = None; cross = []
        for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
            r = cr[i] / (cr[i] - cr[i + 1]); pm = p[i] + r * (p[i + 1] - p[i]); fr = int(round(f[i] + r * (f[i + 1] - f[i])))
            u = np.dot(pm - self.LA, self.AB) / np.dot(self.AB, self.AB)
            if not (self.u_range[0] <= u <= self.u_range[1]): continue
            if last is not None and (fr - last) / self.fps < self.min_gap: continue
            last = fr; cross.append((fr, "load" if s[i] > 0 else "unload"))
        net = "load" if s[0] > 0 else "unload"
        hit = [x[0] for x in cross if x[1] == net] if s[0] != s[-1] else []
        if hit: return hit[-1], net
        # 기준선 끝이 화면 밖으로 연장된 경우: 선을 넘기 전에 화면 가장자리로 잘려 나가는 근접 차량
        if allow_edge and min(self.LA[0], self.LB[0]) < 0:
            x1 = np.concatenate([T[t]["x1"] for t in c])[o]
            if x1[-1] <= self.edge_x and s[-1] > 0 and p[-1, 0] < p[0, 0]: return int(f[-1]), "load"
            if x1[0] <= self.edge_x and s[0] > 0 and p[-1, 0] > p[0, 0]: return int(f[0]), "unload"
        return None

    def emit(self, c, fr, op, now_f, final):
        name = "|".join(map(str, c))
        with self.lock:
            for e in self.events:  # 같은 방향 dup_gap 이내 → 같은 차(박스 2개)
                if e["op"] == op and abs(fr - e["frame"]) / self.fps < self.dup_gap:
                    for t in c: self.emitted[t] = e
                    return
            self.n_events += 1
            ev = dict(no=self.n_events, frame=int(fr), op=op, emit_frame=int(now_f), emit_wall=time.time(),
                      decision=None, reason="", marks={}, mark_s=None, decided_wall=None, n_crops=None, snapshot=None,
                      boxes=self.chain_boxes(c))
            self.events.append(ev)
        for t in c: self.emitted[t] = ev
        cs = self.crops_for(c)
        if cs: ev["snapshot"] = cs[len(cs) // 2]["img"]
        if self.on_event: self.on_event(ev)
        if final or len(cs) >= self.min_crops: self.submit(ev, cs)
        else: self.pending.append((ev, c, now_f + self.grace * self.fps))

    def submit(self, ev, cs):
        ev["n_crops"] = len(cs)
        if cs and ev["snapshot"] is None: ev["snapshot"] = cs[len(cs) // 2]["img"]
        self.mw.q.put((ev, cs, self._decided))

    def _decided(self, ev, dec, reason, marks, dt):
        with self.lock: ev.update(decision=dec, reason=reason, marks=marks, mark_s=round(dt, 3), decided_wall=time.time())
        if self.on_decision: self.on_decision(ev)

    def evaluate(self, now_f, flush=False):
        df = self.pieces_df()
        if df.empty: return
        T = track_table(df, self.min_h)
        chains = build_chains(T, self.max_gap * self.fps, self.link_r, (self.side, self.occl_gap * self.fps, self.occl_r), self.still, self.occl_min_dx)
        wait_f = (max(self.max_gap, self.occl_gap) + self.finalize) * self.fps
        alive = set(df.tid.unique())
        for c in chains:
            done = [self.emitted[t] for t in c if t in self.emitted]
            if done:
                for t in c:
                    if t not in self.emitted: self.emitted[t] = done[0]
            else:
                x = self.crossing(c, T, allow_edge=False)
                if x: self.emit(c, x[0], x[1], now_f, final=False)
            f1 = max(T[t]["f1"] for t in c)
            if not flush and now_f - f1 < wait_f: continue
            if not flush and any(self.raw[t % 1000000].rows[-1][0] > now_f - wait_f and self.raw[t % 1000000].pid(self.raw[t % 1000000].k) == t for t in c): continue
            for q in list(self.pending):
                if any(t in c for t in q[1]): self.pending.remove(q); self.submit(q[0], self.crops_for(q[1]))
            if not any(t in self.emitted for t in c):
                x = self.crossing(c, T, allow_edge=True)
                if x: self.emit(c, x[0], x[1], now_f, final=True)
            self.consumed.update(c)
        for q in list(self.pending):
            ev, c, dl = q; cs = self.crops_for(c)
            if flush or len(cs) >= self.min_crops or now_f >= dl: self.pending.remove(q); self.submit(ev, cs)
        for pid in alive - set(T):
            rt = self.raw[pid % 1000000]
            if flush or rt.rows[-1][0] < now_f - wait_f: self.consumed.add(pid)
        for tid in [t for t, rt in self.raw.items() if all(rt.pid(k) in self.consumed for k in set(rt.piece))]:
            del self.raw[tid]
        # 오래된 기록 정리 (장시간 운용 시 메모리)
        if len(self.events) > 5000:
            with self.lock:
                old = self.events[:-1000]; self.events = self.events[-1000:]
            ids = {id(e) for e in old}
            self.emitted = {k: v for k, v in self.emitted.items() if id(v) not in ids}
            self.consumed = {p for p in self.consumed if (p % 1000000) in self.raw}

    def reset_tracks(self):
        """스트림 재접속 등으로 시간축이 끊겼을 때 진행 중인 트랙을 정리한다 (센 이벤트는 유지)."""
        self.evaluate(10**12, flush=True)
        # ByteTrack reset() 은 id 를 1부터 다시 매긴다 → 옛 id 로 기억해 둔 '이미 센 조각'을 지우지 않으면
        # 새 차가 같은 id 를 받았을 때 이미 센 차로 보고 세지 않는다.
        self.raw.clear(); self.consumed.clear(); self.pending.clear(); self.emitted.clear()
        try: self.det.predictor.trackers[0].reset()
        except Exception: pass
        self.last_fi = None

    def close(self):
        self.evaluate(10**12, flush=True)
        self.mw.q.put(None); self.mw.join()  # 남은 판정을 끝까지 (시간 제한 없음 — 판정이 빠지면 DB 에 안 남음)
