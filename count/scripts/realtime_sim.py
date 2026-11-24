"""COUNT 실시간(RTSP) 시뮬레이터: 영상 파일을 라이브 스트림처럼 흘려보내며 인과적(online)으로 센다.

- 리더 스레드가 프레임을 실제 pts 시각에 맞춰 '도착'시키고, 크기 1 최신프레임 버퍼에 넣는다(실제 RTSP 그래버와 같음).
  처리가 늦으면 이전 프레임은 버려진다(drop 집계). --no-drop 이면 모든 프레임을 처리하고 실시간 배율만 잰다.
- 1단계: yolo26m + ByteTrack (extract_tracks.py 와 같은 설정: imgsz 640, conf 0.2, COCO car/bus/truck, half).
- 이벤트: count_events.py 의 ID 스위치 분리·조각 연결(occl 포함)·통과 판정을 온라인으로. 체인의 마지막 조각이
  끝나고 --finalize 초가 지나 더는 이어질 수 없을 때 확정한다.
- 2단계: 트랙마다 원본 해상도 크롭을 간격을 늘려가며 보관(extract_mark_crops.py 와 같은 여백) → 확정 시 체인에서 K장을
  고르게 골라 표식 모델(별도 스레드) → conf>=0.5 표식이 >=2 크롭이면 제외.
  세 표식 모두 차량과 같이 움직일 때만 인정(motion_static, merge_marks.py 와 동일). 화면 좌표에 고정된 불빛·글자는 버린다.
- --decide-at cross(기본): 선을 넘는 순간 이벤트를 내고 그때까지 모인 크롭으로 판정(모자라면 --cross-grace 대기).
- 출력: <out>/events.csv, stats.json, report.html (+ --gt 가 있으면 eval)

홀드아웃('검증용')은 --final-holdout 없이는 거부한다.
사용: python count/scripts/realtime_sim.py data/COUNT/260818_s1_rtspcam.mp4 [--max-seconds 600] [--gt data/COUNT/gt_rev/260818_s1_rtspcam.csv]
"""
import argparse, csv, json, queue, sys, threading, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_events import frame_pts, track_table, build_chains  # noqa: E402

VEHICLE = [2, 5, 7]
CLS = {0: "plate", 1: "beacon", 2: "number"}


# ---------------------------------------------------------------- 스트림 리더
class Reader(threading.Thread):
    def __init__(self, path, pts, no_drop, max_frames, start_frame=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path)); self.pts = pts; self.no_drop = no_drop; self.max_frames = max_frames; self.start_frame = start_frame
        self.q = queue.Queue(maxsize=64) if no_drop else None
        self.lock = threading.Lock(); self.latest = None; self.ev = threading.Event()
        self.received = 0; self.dropped = 0; self.done = False; self.t0 = None; self.late = []; self.error = None

    def run(self):
        try: self._run()
        except BaseException as e:  # 리더가 죽으면 처리 루프가 영원히 기다리지 않게 종료 신호를 보낸다
            self.error = repr(e); self.done = True; self.ev.set()
            if self.no_drop: self.q.put(None)
            raise

    def _run(self):
        fi = -1
        for _ in range(self.start_frame):  # --start: 앞부분은 디코딩만 하고 버림 (실제 pts 기준 위치)
            if not self.cap.grab(): break
            fi += 1
        self.t0 = time.perf_counter(); p0 = self.pts[min(self.start_frame, len(self.pts) - 1)]
        while True:
            fi += 1
            if self.max_frames and fi >= self.max_frames: break
            ok, img = self.cap.read()
            if not ok: break
            arr = self.t0 + (self.pts[min(fi, len(self.pts) - 1)] - p0)  # 이 프레임이 카메라에서 도착하는 벽시계 시각
            if self.no_drop:
                self.q.put((fi, img, time.perf_counter()))
            else:
                now = time.perf_counter()
                if arr > now: time.sleep(arr - now)
                else: self.late.append(now - arr)  # 디코딩이 실시간을 못 따라감
                with self.lock:
                    if self.latest is not None: self.dropped += 1
                    self.latest = (fi, img, max(arr, time.perf_counter()))
                self.ev.set()
            self.received += 1
        self.done = True; self.ev.set()
        if self.no_drop: self.q.put(None)

    def get(self):
        if self.no_drop: return self.q.get()
        while True:
            with self.lock:
                if self.latest is not None:
                    x = self.latest; self.latest = None; self.ev.clear(); return x
                if self.done: return None
            self.ev.wait(0.5)


# ---------------------------------------------------------------- 스트림 재전송(replay) 감지
class ReplayGuard:
    """RTSP 재접속·버퍼 재전송으로 몇 초 전 영상이 새 타임스탬프로 다시 들어오는 경우를 잡는다 (차량 외형 비교 아님).
    160x90 흑백 링버퍼(--replay-buf 초). 장면이 움직일 때(직전 프레임 차 > --replay-motion)만, 0.5초보다 오래된 버퍼 프레임과 비교해
    최소 차이가 min(--replay-abs, --replay-rel × 직전 차) 미만인 프레임이 --replay-k 장 연속이면 재전송 구간 시작,
    맞지 않는 프레임이 2장 연속이면 끝. 이 장면은 같은 차종이 2초 간격으로 같은 자리를 지나 '비슷한' 프레임이 흔하므로
    (260818_s1 에서 0.75×중앙값 기준이면 오검출 51건) 재인코딩 수준의 거의 같은 프레임만 인정한다."""
    def __init__(self, fps, buf_s, min_age_s, motion, rel, absd, k):
        self.n = max(2, int(buf_s * fps)); self.min_age = max(1, int(min_age_s * fps)); self.motion = motion; self.rel = rel; self.abs = absd; self.k = k
        self.B = np.zeros((self.n, 90, 160), np.float32); self.cnt = 0; self.prev = None; self.run = 0; self.miss = 0; self.on = False; self.cur = None; self.segments = []; self.cost = []

    def check(self, img, fi, t):
        t0 = time.perf_counter()
        sub = img[::4, ::4]  # 싼 1차 축소(뷰) 후 면적 평균
        g = cv2.cvtColor(cv2.resize(sub, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mot = float(np.abs(g - self.prev).mean()) if self.prev is not None else 0.0
        match = False; best = None; off = None
        m = min(self.cnt, self.n)
        if (mot > self.motion or self.on) and m > self.min_age:  # 재전송 중에는 움직임이 적어도 계속 비교
            ages = np.arange(self.min_age, m + 1)  # 몇 프레임 전인지 (min_age..m)
            idx = (self.cnt - ages) % self.n
            d = np.abs(self.B[idx] - g).mean((1, 2)); j = int(d.argmin()); best = float(d[j]); off = int(ages[j])
            match = best < (self.abs if self.on else min(self.abs, self.rel * mot))
            self.last = (mot, best)
        if match:
            self.run += 1; self.miss = 0
            if not self.on and self.run >= self.k:
                self.on = True; self.cur = dict(start_frame=fi, start_s=round(t, 2), offset_frames=off, best_diff=round(best, 3))
        else:
            self.run = 0
            if self.on:
                self.miss += 1
                if self.miss >= 2:
                    self.on = False; self.cur.update(end_frame=fi, end_s=round(t, 2)); self.segments.append(self.cur); self.cur = None
        self.prev = g; self.B[self.cnt % self.n] = g; self.cnt += 1
        self.cost.append(time.perf_counter() - t0)
        return self.on


# ---------------------------------------------------------------- 데모 영상
class Renderer:
    """처리한 프레임마다 기준선·차량 박스(판정 상태 색)·표식 박스·누적 대수 HUD 를 그려 H.264 mp4 로 저장."""
    KO = {"plate": "번호판", "beacon": "경광등", "number": "번호"}
    MC = {0: (230, 40, 40), 1: (255, 176, 0), 2: (225, 40, 225)}  # RGB: 번호판 빨강, 경광등 주황, 번호 보라

    def __init__(self, a, W, H, fps, LA, LB, pts):
        import subprocess
        from PIL import ImageFont
        self.a = a; self.s = a.render_width / W; self.ow = a.render_width; self.oh = int(round(H * self.s / 2) * 2); self.pts = pts
        self.LA = LA * self.s; self.LB = LB * self.s; self.mark = YOLO(a.mark_model); self.min_h = a.min_h
        f = "C:/Windows/Fonts/malgunbd.ttf" if Path("C:/Windows/Fonts/malgunbd.ttf").exists() else "C:/Windows/Fonts/malgun.ttf"
        self.f_big = ImageFont.truetype(f, 30); self.f_sm = ImageFont.truetype(f, 17); self.f_xs = ImageFont.truetype(f, 14)
        Path(a.render).parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.ow}x{self.oh}", "-r", f"{fps:.3f}",
                                   "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.render_crf), "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", str(a.render)], stdin=subprocess.PIPE)
        self.prev = None; self.flash = {}; self.n = 0; self.t_first = None; self.final = None

    @staticmethod
    def counts(evs):
        return dict(load=sum(e["op"] == "load" and e["decision"] == "cargo" for e in evs),
                    unload=sum(e["op"] == "unload" and e["decision"] == "cargo" for e in evs),
                    excl=sum(e["decision"] == "exclude" for e in evs))

    def frame(self, fi, img, vs, evs):
        from PIL import Image, ImageDraw
        t = float(self.pts[min(fi, len(self.pts) - 1)])
        if self.a.render_from is not None and t < self.a.render_from: return
        if self.t_first is None: self.t_first = t; self.base = self.counts(evs)
        evs_c = self.counts(evs); evs_c = {k: evs_c[k] - self.base[k] for k in evs_c}
        small = cv2.resize(img, (self.ow, self.oh), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im, "RGBA"); s = self.s
        d.line([tuple(self.LA), tuple(self.LB)], fill=(0, 230, 255, 230), width=3)
        # 표식: 큰 차량 박스 크롭에만 (데모용, 매 프레임)
        crops, offs = [], []
        for _, (x1, y1, x2, y2), _ in vs:
            if y2 - y1 < self.min_h: continue
            w, h = x2 - x1, y2 - y1
            cx1, cy1 = max(0, int(x1 - 0.06 * w)), max(0, int(y1 - 0.3 * h)); cx2, cy2 = min(img.shape[1], int(x2 + 0.06 * w)), int(y2)
            crops.append(img[cy1:cy2, cx1:cx2]); offs.append((cx1, cy1))
        if crops:
            for (ox, oy), res in zip(offs, self.mark.predict(crops, imgsz=640, conf=self.a.veh_conf, half=True, verbose=False)):
                for (mx1, my1, mx2, my2), k in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist()):
                    d.rectangle([(ox + mx1) * s, (oy + my1) * s, (ox + mx2) * s, (oy + my2) * s], outline=self.MC[k], width=2)
        for tid, (x1, y1, x2, y2), ev in vs:
            if ev is None or ev.get("decision") is None:
                col = (170, 175, 180); lab = f"#{tid}" + (" 판정 중" if ev is not None else "")
            elif ev["decision"] == "cargo":
                col = (40, 200, 90); lab = f"#{tid} 화물"
            else:
                col = (235, 60, 50); mk = ev.get("marks") or {}
                lab = f"#{tid} 제외 " + "·".join(self.KO[k] for k in ("plate", "beacon", "number") if mk.get(k))
            b = [x1 * s, y1 * s, x2 * s, y2 * s]; d.rectangle(b, outline=col + (255,), width=3)
            tw = d.textlength(lab, font=self.f_sm); ty = max(0, b[1] - 24)
            d.rectangle([b[0], ty, b[0] + tw + 10, ty + 23], fill=col + (225,)); d.text((b[0] + 5, ty + 1), lab, font=self.f_sm, fill=(255, 255, 255))
        c = evs_c
        if self.prev is not None:
            for k in c:
                if c[k] != self.prev[k]: self.flash[k] = self.n + 15
        self.prev = c
        d.rounded_rectangle([12, 12, 352, 132], 10, fill=(15, 20, 26, 190))
        rows = [("선적 화물", "load", (120, 200, 255)), ("하역 화물", "unload", (120, 200, 255)), ("제외 (세이프티·작업)", "excl", (255, 150, 120))]
        for i, (name, k, cc) in enumerate(rows):
            y = 20 + i * 36; on = self.flash.get(k, -1) > self.n
            if on: d.rounded_rectangle([18, y - 3, 346, y + 33], 7, fill=(255, 210, 60, 225))
            d.text((28, y + 5), name, font=self.f_sm, fill=(20, 20, 20) if on else (220, 226, 232))
            d.text((334, y - 3), str(c[k]), font=self.f_big, anchor="ra", fill=(20, 20, 20) if on else cc)
        clk = f"영상 {int(t // 60):02d}:{t % 60:04.1f}  ·  경과 {t - self.t_first:4.1f}s"
        d.rounded_rectangle([12, 140, 352, 166], 8, fill=(15, 20, 26, 170)); d.text((22, 143), clk, font=self.f_xs, fill=(220, 226, 232))
        lg = [("화물", (40, 200, 90)), ("제외", (235, 60, 50)), ("판정 전", (170, 175, 180)), ("번호판", self.MC[0]), ("경광등", self.MC[1]), ("번호", self.MC[2])]
        x = 12; y = self.oh - 32
        d.rounded_rectangle([8, y - 4, 14 + sum(d.textlength(n, font=self.f_xs) + 34 for n, _ in lg), y + 24], 8, fill=(15, 20, 26, 170))
        for n, cc in lg:
            d.rectangle([x + 6, y + 4, x + 20, y + 18], outline=cc, width=3); d.text((x + 25, y + 2), n, font=self.f_xs, fill=(230, 234, 238))
            x += d.textlength(n, font=self.f_xs) + 34
        if self.a.render_banner:
            bw_ = d.textlength(self.a.render_banner, font=self.f_sm)
            x0 = max(360, (self.ow - bw_) / 2 - 14)
            d.rounded_rectangle([x0, 12, x0 + bw_ + 28, 44], 8, fill=(190, 30, 30, 220)); d.text((x0 + 14, 17), self.a.render_banner, font=self.f_sm, fill=(255, 255, 255))
        if self.a.render_title:
            tw = d.textlength(self.a.render_title, font=self.f_sm)
            d.rounded_rectangle([self.ow - tw - 34, 12, self.ow - 12, 44], 8, fill=(15, 20, 26, 190))
            d.text((self.ow - tw - 23, 17), self.a.render_title, font=self.f_sm, fill=(255, 255, 255))
        self.p.stdin.write(np.asarray(im).tobytes()); self.n += 1

    def close(self, evs):
        self.p.stdin.close(); self.p.wait(); self.final = self.counts(evs)
        print("render:", self.a.render, self.final, f"{Path(self.a.render).stat().st_size / 2**20:.1f} MB", flush=True)


# ---------------------------------------------------------------- 표식 판정
def motion_static(D):
    """세 표식 모두: 차량이 실제로 움직인 크롭 쌍(박스 x1,x2 가 같은 방향으로 0.25w 이상)에서
    화면 좌표가 고정(<0.04w)인데 차량 기준 같은 자리(rx,ry 차 <0.10)로 따라 움직인 짝이 없으면 고정물로 버린다 (merge_marks.py 와 동일).
    한 크롭에만 있는 표식은 비교 대상이 없으므로 유지."""
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


def beacon_ok_idx(D):
    """예전 규칙(경광등만): 다른 크롭에 상대 위치 비슷한(<0.12) 경광등이 있고 화면 고정이 아니면 인정. 인정된 D 인덱스."""
    ok = set(); B = [n for n, d in enumerate(D) if d[1] == 1]
    for i in B:
        for j in B:
            a, b = D[i], D[j]
            if a[0] == b[0]: continue
            rel = np.hypot(a[8] - b[8], a[9] - b[9]); w = max(a[7], b[7])
            veh_move = abs((a[5] + a[6]) / 2 - (b[5] + b[6]) / 2)
            static = np.hypot(a[3] - b[3], a[4] - b[4]) < 0.05 * w and veh_move > 0.2 * w
            if rel < 0.12 and not static: ok.add(i); ok.add(j)
    return ok


class MarkWorker(threading.Thread):
    def __init__(self, model, conf, min_frames, imgsz=640):
        super().__init__(daemon=True)
        self.m = YOLO(model); self.conf = conf; self.min_frames = min_frames; self.imgsz = imgsz
        self.q = queue.Queue(); self.out = []; self.lat = []; self.motion = "all"; self.dropped = {v: 0 for v in CLS.values()}; self.mlog = []

    def run(self):
        while True:
            job = self.q.get()
            if job is None: break
            ev, crops, cb = job; t = time.perf_counter()
            res = self.m.predict([c["img"] for c in crops], imgsz=self.imgsz, conf=0.25, half=True, verbose=False) if crops else []
            D = []  # (crop_i, cls, conf, gx, gy, bx1, bx2, vw, rx, ry)
            for i, (c, r) in enumerate(zip(crops, res)):
                bw = c["bx2"] - c["bx1"]; bh = c["by2"] - c["by1"]
                for (x1, y1, x2, y2), cf, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
                    if cf < self.conf: continue
                    gx = c["cx1"] + (x1 + x2) / 2; gy = c["cy1"] + (y1 + y2) / 2
                    D.append((i, k, cf, gx, gy, c["bx1"], c["bx2"], bw, (gx - c["bx1"]) / bw, (gy - c["by1"]) / bh))
            static = motion_static(D) if self.motion == "all" else set()
            old_static = {n for n, d in enumerate(D) if d[1] == 1} - beacon_ok_idx(D)  # 예전 규칙(경광등만, 짝 확인 필요) 비교용
            def decide(drop):
                hits = {}
                for n, d in enumerate(D):
                    if n not in drop: hits.setdefault(d[0], set()).add(CLS[d[1]])
                return hits
            hits = decide(static); hits_old = decide(old_static)
            nf = len(hits); excl = nf >= self.min_frames; excl_old = len(hits_old) >= self.min_frames
            dropped = {CLS[k]: sum(1 for n in static if D[n][1] == k) for k in CLS}
            for k, v in dropped.items(): self.dropped[k] += v
            cnt = {k: sum(k in s for s in hits.values()) for k in ("plate", "beacon", "number")}
            dd = " ".join(f"{k}{v}" for k, v in dropped.items() if v)
            reason = " ".join(f"{k}{v}" for k, v in cnt.items() if v) + (f" (고정 표식 버림 {dd})" if dd else "")
            ev["decision_old_rule"] = "exclude" if excl_old else "cargo"; ev["static_dropped"] = dd; ev["marks"] = cnt
            # 기록 전용(판정에 영향 없음): 크롭별·종류별 최대 conf — raw(≥0.25, 이동 규칙 전) / kept(≥veh_conf, 이동 규칙 후) / static(버린 수)
            for i, (c, r) in enumerate(zip(crops, res)):
                row = dict(ev=ev, crop=i, frame=c["frame"])
                for k, nm in CLS.items():
                    cf = [x for x, kk in zip(r.boxes.conf.tolist(), r.boxes.cls.int().tolist()) if kk == k]
                    row[f"{nm}_raw"] = round(max(cf), 3) if cf else 0.0
                    kept = [D[n][2] for n in range(len(D)) if D[n][0] == i and D[n][1] == k and n not in static]
                    row[f"{nm}_kept"] = round(max(kept), 3) if kept else 0.0
                    row[f"{nm}_static"] = sum(1 for n in static if D[n][0] == i and D[n][1] == k)
                self.mlog.append(row)
            dt = time.perf_counter() - t; self.lat.append(dt)
            cb(ev,"exclude" if excl else "cargo", f"{nf}/{len(crops)} 크롭: {reason}".strip(), dt)


# ---------------------------------------------------------------- 온라인 트랙/체인
class RawTrack:
    """ByteTrack 원본 tid 하나. split_id_switch 를 증분으로 적용해 조각(piece) 번호를 매긴다."""
    def __init__(self, tid): self.tid = tid; self.rows = []; self.piece = []; self.d = 0; self.ext = 0; self.k = 0; self.crops = {}; self.stride = 1; self.nc = 0

    def add(self, row, thr):
        self.rows.append(row); self.piece.append(self.k); x = (row[3] + row[5]) / 2; i = len(self.rows) - 1
        xs = lambda j: (self.rows[j][3] + self.rows[j][5]) / 2
        if i == 0: return
        if self.d == 0:
            if abs(x - xs(self.ext)) >= thr: self.d = int(np.sign(x - xs(self.ext))); self.ext = i
        elif (x - xs(self.ext)) * self.d > 0: self.ext = i
        elif abs(x - xs(self.ext)) >= thr:
            self.k += 1
            for j in range(self.ext + 1, i + 1): self.piece[j] = self.k
            self.d = -self.d; self.ext = i

    def pid(self, k): return self.tid + 1000000 * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--line-key", default=None, help="data/COUNT/lines.json 키 (기본: 영상 이름)")
    ap.add_argument("--model", default=str(ROOT / "_model_/count/yolo26m.pt"))
    ap.add_argument("--mark-model", default=str(ROOT / "runs/count/marks/r1/weights/best.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--no-drop", action="store_true", help="모든 프레임 처리(실시간 배율만 측정)")
    ap.add_argument("--max-seconds", type=float, default=0, help="이 길이(초)만 처리")
    ap.add_argument("--start", type=float, default=0, help="시작 위치(초, 실제 pts)")
    ap.add_argument("--render", default=None, help="데모 영상 출력 경로(.mp4). --no-drop 과 함께 쓰는 것을 권장")
    ap.add_argument("--render-width", type=int, default=1280); ap.add_argument("--render-crf", type=int, default=26)
    ap.add_argument("--render-title", default=None, help="데모 우상단 제목(출처 영상·시각)")
    ap.add_argument("--render-from", type=float, default=None, help="이 시각(초)부터만 영상에 기록 (앞부분은 추적 상태를 맞추려고 처리만)")
    ap.add_argument("--render-banner", default=None, help="상단 중앙 배너 문구 (오류 클립용)")
    # count_events.py 와 동일한 기본값
    ap.add_argument("--min-h", type=float, default=150); ap.add_argument("--split-x", type=float, default=250)
    ap.add_argument("--edge-x", type=float, default=40); ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--max-gap", type=float, default=1.5); ap.add_argument("--link-r", type=float, default=300)
    ap.add_argument("--min-travel", type=float, default=300); ap.add_argument("--dup-gap", type=float, default=0.6)
    ap.add_argument("--occl-gap", type=float, default=1.0); ap.add_argument("--occl-r", type=float, default=800)
    ap.add_argument("--u-range", type=float, nargs=2, default=[-0.5, 1.1])
    ap.add_argument("--finalize", type=float, default=0.7, help="max(max-gap, occl-gap) 뒤 추가 대기(초) 후 체인 확정")
    ap.add_argument("--decide-at", choices=["cross", "finalize"], default="cross", help="cross: 선을 넘는 순간 이벤트+판정 / finalize: 체인 확정 후(예전 방식)")
    ap.add_argument("--min-crops", type=int, default=3, help="cross 모드: 통과 시점 크롭이 이보다 적으면 grace 동안 기다림")
    ap.add_argument("--cross-grace", type=float, default=0.5, help="cross 모드: 크롭 대기 최대(초)")
    ap.add_argument("--replay", choices=["skip", "log", "off"], default="skip", help="스트림 재전송 감지: skip=그 구간 추적·카운트 제외, log=기록만")
    ap.add_argument("--replay-buf", type=float, default=5.0); ap.add_argument("--replay-min-age", type=float, default=0.5)
    ap.add_argument("--replay-motion", type=float, default=1.0, help="직전 프레임 차(0~255 평균)가 이보다 클 때만 비교")
    ap.add_argument("--replay-rel", type=float, default=0.2); ap.add_argument("--replay-abs", type=float, default=0.6)
    ap.add_argument("--replay-k", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=1, help="처리 프레임 몇 장마다 통과 판정 (finalize 모드는 0.5초)")
    # 표식
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--pad-top", type=float, default=0.3); ap.add_argument("--pad-x", type=float, default=0.06)
    ap.add_argument("--keep-crops", type=int, default=24, help="원본 트랙당 보관 크롭 상한(넘으면 간격 2배)")
    ap.add_argument("--veh-conf", type=float, default=0.5); ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--gt", default=None, help="gt_rev csv (있으면 이벤트 매칭 평가)")
    ap.add_argument("--final-holdout", action="store_true", help="검증용 영상 최종 평가 때만")
    a = ap.parse_args()

    vid = Path(a.video)
    if "검증용" in str(vid) and not a.final_holdout:
        raise SystemExit("홀드아웃(검증용) 영상은 --final-holdout 없이 실행하지 않습니다")
    key = a.line_key or vid.stem.replace("_검증용", "")
    lines = json.load(open(ROOT / "data/COUNT/lines.json"))
    if key not in lines: raise SystemExit(f"{key}: lines.json 에 기준선 없음 (--line-key)")
    LA, LB = np.array(lines[key], float); AB = LB - LA
    side = lambda q: AB[0] * (q[1] - LA[1]) - AB[1] * (q[0] - LA[0]) >= 0
    out = Path(a.out) if a.out else ROOT / "runs/count/rt" / vid.stem; out.mkdir(parents=True, exist_ok=True)

    pts = frame_pts(str(vid.resolve()), out / "pts.npy")
    cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    start_f = int(np.searchsorted(pts - pts[0], a.start)) if a.start else 0
    max_frames = int(np.searchsorted(pts - pts[0], a.start + a.max_seconds)) if a.max_seconds else 0
    dur = float(pts[min(len(pts), max_frames or len(pts)) - 1] - pts[start_f])
    print(f"{vid.name}: {fps:.2f}fps {W}x{H}, 길이 {dur/60:.1f}분, mode={'no-drop' if a.no_drop else 'realtime'}", flush=True)

    det = YOLO(a.model)
    det.predict(np.zeros((H, W, 3), np.uint8), imgsz=a.imgsz, half=True, verbose=False)  # 워밍업 (추적기는 첫 track 호출에서 새로 생성)
    mw = MarkWorker(a.mark_model, a.veh_conf, a.min_frames)
    mw.m.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, half=True, verbose=False); mw.start()
    torch.cuda.reset_peak_memory_stats()

    raw = {}; consumed = set(); events = []; ev_lock = threading.Lock(); merged = []
    det_lat = []; age = []; arrive_wall = {}

    def wall_of(fr):  # 그 프레임이 카메라에서 도착한 벽시계 시각(리더 시작 기준)
        return reader.t0 + (pts[min(fr, len(pts) - 1)] - pts[start_f])

    def on_decision(ev, dec, reason, dt):
        with ev_lock:
            ev.update(decision=dec, reason=reason, mark_s=round(dt, 3), decided_wall=time.perf_counter())

    def pieces_df():
        rows = []
        for rt in raw.values():
            n = len(rt.rows)
            # 오래 머문 트랙(주차 차량 등)은 앞 10개·뒤 150개만 전부, 중간은 최대 100개로 고르게 (주차 차량이 많으면 매 프레임 비용이 계속 커짐) (통과 판정·속도 추정에 필요한 양 끝 유지)
            sel = range(n) if n <= 300 else [*range(10), *np.linspace(10, n - 151, 100).astype(int).tolist(), *range(n - 150, n)]
            for q in sel:
                r, k = rt.rows[q], rt.piece[q]
                pid = rt.pid(k)
                if pid in consumed: continue
                rows.append((r[0], pid, r[1], r[2], r[3], r[4], r[5], r[6]))
        return pd.DataFrame(rows, columns=["frame", "tid", "cls", "conf", "x1", "y1", "x2", "y2"])

    def crops_for(chain):
        cs = []
        for pid in chain:
            rt = raw.get(pid % 1000000)
            if rt is None: continue
            k = pid // 1000000
            cs += [c for j, c in rt.crops.items() if rt.piece[j] == k]
        cs.sort(key=lambda c: c["frame"])
        if len(cs) > a.k: cs = [cs[i] for i in np.unique(np.linspace(0, len(cs) - 1, a.k).round().astype(int))]
        return cs

    emitted = {}       # 조각 id → 이미 센 이벤트 (cross 모드: 나중에 이어 붙은 조각이 있어도 다시 세지 않음)
    pending_dec = []   # (event, chain, 마감 프레임): 크롭이 모자라 grace 동안 기다리는 판정
    eval_lat = []

    def crossing(c, T, allow_edge):
        """count_events.py 와 같은 통과 판정. 지금까지의 궤적으로 (frame, dir) 또는 None."""
        f = np.concatenate([T[t]["f"] for t in c]); p = np.concatenate([T[t]["p"] for t in c])
        o = np.argsort(f, kind="stable"); f, p = f[o], p[o]
        if np.ptp(p[:, 0]) < a.min_travel: return None
        cr = AB[0] * (p[:, 1] - LA[1]) - AB[1] * (p[:, 0] - LA[0]); s = np.where(cr >= 0, 1, -1); last = None; cross = []
        for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
            r = cr[i] / (cr[i] - cr[i + 1]); pm = p[i] + r * (p[i + 1] - p[i]); fr = int(round(f[i] + r * (f[i + 1] - f[i])))
            u = np.dot(pm - LA, AB) / np.dot(AB, AB)
            if not (a.u_range[0] <= u <= a.u_range[1]): continue
            if last is not None and (fr - last) / fps < a.min_gap: continue
            last = fr; cross.append((fr, "R2L" if s[i] > 0 else "L2R"))
        net = "R2L" if s[0] > 0 else "L2R"; hit = [x[0] for x in cross if x[1] == net] if s[0] != s[-1] else []
        if hit: return hit[-1], net
        if allow_edge and LA[0] < 0:
            x1 = np.concatenate([T[t]["x1"] for t in c])[o]
            if x1[-1] <= a.edge_x and s[-1] > 0 and p[-1, 0] < p[0, 0]: return int(f[-1]), "R2L"
            if x1[0] <= a.edge_x and s[0] > 0 and p[-1, 0] > p[0, 0]: return int(f[0]), "L2R"
        return None

    def emit(c, fr, dr, now_f, final):
        name = "|".join(map(str, c))
        with ev_lock:
            for e in events:  # 같은 방향 dup-gap 이내 → 같은 차(박스 2개)
                if e["dir"] == dr and abs(fr - e["frame"]) / fps < a.dup_gap:
                    e["chain"] += "|" + name; merged.append(name)
                    for t in c: emitted[t] = e
                    return
            ev = dict(event_no=len(events), frame=int(fr), dir=dr, chain=name, emit_frame=int(now_f), emit_wall=time.perf_counter(),
                      decision=None, reason="", mark_s=None, decided_wall=None, n_crops=None, late=final and a.decide_at == "cross")
            events.append(ev)
        for t in c: emitted[t] = ev
        cs = crops_for(c)
        if final or a.decide_at == "finalize" or len(cs) >= a.min_crops: submit(ev, cs)
        else: pending_dec.append((ev, c, now_f + a.cross_grace * fps))

    def submit(ev, cs):
        ev["n_crops"] = len(cs); mw.q.put((ev, cs, on_decision))

    def evaluate(now_f, flush=False):
        te = time.perf_counter()
        df = pieces_df()
        if df.empty: return
        T = track_table(df, a.min_h)
        occl = (side, a.occl_gap * fps, a.occl_r) if a.occl_gap > 0 else None
        chains = build_chains(T, a.max_gap * fps, a.link_r, occl)
        wait_f = (max(a.max_gap, a.occl_gap) + a.finalize) * fps
        alive = {pid for pid in df.tid.unique()}
        for c in chains:
            done = [emitted[t] for t in c if t in emitted]
            if done:  # 이미 센 체인에 새 조각이 붙음 → 다시 세지 않음
                for t in c:
                    if t not in emitted: emitted[t] = done[0]; done[0]["chain"] += "|" + str(t)
            elif a.decide_at == "cross":
                x = crossing(c, T, allow_edge=False)
                if x: emit(c, x[0], x[1], now_f, final=False)
            f1 = max(T[t]["f1"] for t in c)
            if not flush and now_f - f1 < wait_f: continue
            # 같은 원본 tid 의 조각이 아직 진행 중이면(마지막 검출이 최근) 보류
            if not flush and any(raw[t % 1000000].rows[-1][0] > now_f - wait_f and raw[t % 1000000].pid(raw[t % 1000000].k) == t for t in c): continue
            # 확정: 아직 안 센 체인만 (cross 모드에선 화면 끝 통과 등 늦게 성립한 경우)
            for q in list(pending_dec):
                if any(t in c for t in q[1]): pending_dec.remove(q); submit(q[0], crops_for(q[1]))
            if not any(t in emitted for t in c):
                x = crossing(c, T, allow_edge=True)
                if x: emit(c, x[0], x[1], now_f, final=True)
            consumed.update(c)
        # grace 가 끝났거나 크롭이 모인 판정 보내기
        for q in list(pending_dec):
            ev, c, dl = q; cs = crops_for(c)
            if flush or len(cs) >= a.min_crops or now_f >= dl: pending_dec.remove(q); submit(ev, cs)
        # 체인에 못 들어간 조각(검출 2개 미만, min_h 미달)도 오래되면 정리
        for pid in alive - set(T):
            rt = raw[pid % 1000000]
            if flush or rt.rows[-1][0] < now_f - wait_f: consumed.add(pid)
        for tid in [t for t, rt in raw.items() if all(rt.pid(k) in consumed for k in set(rt.piece))]:
            del raw[tid]
        eval_lat.append(time.perf_counter() - te)

    rg = None if a.replay == "off" else ReplayGuard(fps, a.replay_buf, a.replay_min_age, a.replay_motion, a.replay_rel, a.replay_abs, a.replay_k)
    replay_skipped = 0
    rd = Renderer(a, W, H, fps, LA, LB, pts) if a.render else None
    reader = Reader(vid, pts, a.no_drop, max_frames, start_f); reader.start()
    while reader.t0 is None: time.sleep(0.01)
    proc = 0; last_eval = -1; tlog = time.perf_counter()
    while True:
        x = reader.get()
        if x is None: break
        fi, img, t_arr = x
        t = time.perf_counter(); age.append(t - t_arr)
        if rg is not None and rg.check(img, fi, float(pts[min(fi, len(pts) - 1)] - pts[0])) and a.replay == "skip":
            replay_skipped += 1; continue
        r = det.track(img, persist=True, imgsz=a.imgsz, conf=a.conf, classes=VEHICLE, tracker=a.tracker, half=True, verbose=False)[0]
        det_lat.append(time.perf_counter() - t); proc += 1
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cl, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                rt = raw.get(tid)
                if rt is None: rt = raw[tid] = RawTrack(tid)
                if rt.pid(rt.k) in consumed and rt.rows: continue  # 이미 확정된 조각이 다시 살아남 → 무시
                rt.add((fi, cl, cf, x1, y1, x2, y2), a.split_x)
                if y2 - y1 >= a.min_h:
                    rt.nc += 1
                    if (rt.nc - 1) % rt.stride == 0:
                        w, h = x2 - x1, y2 - y1
                        cx1, cy1 = max(0, int(x1 - a.pad_x * w)), max(0, int(y1 - a.pad_top * h)); cx2, cy2 = min(W, int(x2 + a.pad_x * w)), min(H, int(y2))
                        rt.crops[len(rt.rows) - 1] = dict(frame=fi, img=img[cy1:cy2, cx1:cx2].copy(), cx1=cx1, cy1=cy1, bx1=x1, by1=y1, bx2=x2, by2=y2)
                        if len(rt.crops) > a.keep_crops:  # 간격 2배로 솎기
                            rt.stride *= 2
                            for j in sorted(rt.crops)[1::2]: del rt.crops[j]
        if (a.decide_at == "cross" and proc % a.eval_every == 0) or fi - last_eval >= fps * 0.5:
            evaluate(fi); last_eval = fi
        if rd is not None:
            vs = []
            if r.boxes.id is not None:
                for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist()):
                    rt = raw.get(tid); pid = rt.pid(rt.k) if rt else tid
                    vs.append((tid, (x1, y1, x2, y2), emitted.get(pid)))
            with ev_lock: evs = [dict(op="load" if e["dir"] == "R2L" else "unload", decision=e["decision"]) for e in events]
            rd.frame(fi, img, vs, evs)
        if time.perf_counter() - tlog > 30:
            tlog = time.perf_counter(); el = tlog - reader.t0
            print(f"  {pts[fi]-pts[0]:7.0f}s  proc {proc} drop {reader.dropped}  {proc/el:.1f}fps  events {len(events)}", flush=True)
    if reader.error: raise SystemExit(f"스트림 리더 오류로 중단: {reader.error}")
    evaluate(10**9, flush=True)
    mw.q.put(None); mw.join()
    if rd is not None:
        with ev_lock: evs = [dict(op="load" if e["dir"] == "R2L" else "unload", decision=e["decision"]) for e in events]
        rd.close(evs)
    t_end = time.perf_counter(); wall = t_end - reader.t0

    # ---------------- 출력
    rows = []
    for e in sorted(events, key=lambda e: e["frame"]):
        ct = float(pts[min(e["frame"], len(pts) - 1)] - pts[0])
        rows.append(dict(event_no=e["event_no"], cross_time=f"{int(ct//60):02d}:{int(ct%60):02d}", cross_s=round(ct, 1), dir=e["dir"],
                         op="load" if e["dir"] == "R2L" else "unload", decision=e["decision"], reason=e["reason"], chain=e["chain"],
                         decided_at=round(e["decided_wall"] - reader.t0, 1), latency_s=round(e["decided_wall"] - wall_of(e["frame"]), 1),
                         mark_s=e["mark_s"], n_crops=e["n_crops"], emit_latency_s=round(e["emit_wall"] - wall_of(e["frame"]), 2), late=bool(e["late"]), decision_old_rule=e.get("decision_old_rule"), static_dropped=e.get("static_dropped", "")))
    ev = pd.DataFrame(rows); ev.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"event_no": r["ev"]["event_no"], "cross_time": next((x["cross_time"] for x in rows if x["event_no"] == r["ev"]["event_no"]), ""),
                   "decision": r["ev"]["decision"], **{k: v for k, v in r.items() if k != "ev"}} for r in mw.mlog]).to_csv(out / "events_marks.csv", index=False, encoding="utf-8-sig")
    q = lambda v, p: round(float(np.percentile(v, p)), 4) if len(v) else None
    cnt = {f"{op}_{d}": int(((ev.op == op) & (ev.decision == d)).sum()) if len(ev) else 0 for op in ("load", "unload") for d in ("cargo", "exclude")}
    st = dict(video=vid.name, mode=("no-drop" if a.no_drop else "realtime"), fps_source=round(fps, 3), duration_s=round(dur, 1), wall_s=round(wall, 1),
              realtime_factor=round(dur / wall, 3), counts=cnt, events=len(ev), merged_dup=len(merged),
              frames_received=reader.received, frames_processed=proc, frames_dropped=reader.dropped, processed_fps=round(proc / wall, 2),
              decode_late_frames=len(reader.late), decode_late_p95_s=q(reader.late, 95),
              det_latency_s=dict(p50=q(det_lat, 50), p95=q(det_lat, 95), max=q(det_lat, 100)),
              frame_age_s=dict(p50=q(age, 50), p95=q(age, 95), max=q(age, 100)),
              mark_latency_s=dict(p50=q(mw.lat, 50), p95=q(mw.lat, 95), max=q(mw.lat, 100)),
              decision_latency_s=dict(p50=q(ev.latency_s, 50), p95=q(ev.latency_s, 95), max=q(ev.latency_s, 100)) if len(ev) else None,
              decide_at=a.decide_at, cross_grace_s=a.cross_grace, min_crops=a.min_crops,
              emit_latency_s=dict(p50=q(ev.emit_latency_s, 50), p95=q(ev.emit_latency_s, 95), max=q(ev.emit_latency_s, 100)) if len(ev) else None,
              decisions_lt_min_crops=int((ev.n_crops < a.min_crops).sum()) if len(ev) else 0,
              crops_hist={int(k): int(v) for k, v in ev.n_crops.value_counts().sort_index().items()} if len(ev) else {},
              late_events=int(ev.late.sum()) if len(ev) else 0,
              replay=dict(mode=a.replay, segments=(rg.segments + ([rg.cur] if rg and rg.cur else [])) if rg else [], frames_skipped=replay_skipped,
                          cost_ms_p50=q([x * 1000 for x in rg.cost], 50) if rg else None, cost_ms_p95=q([x * 1000 for x in rg.cost], 95) if rg else None),
              motion_rule="all", static_marks_dropped=mw.dropped,
              decisions_changed_vs_beacon_rule=int((ev.decision != ev.decision_old_rule).sum()) if len(ev) else 0,
              eval_latency_s=dict(p50=q(eval_lat, 50), p95=q(eval_lat, 95), max=q(eval_lat, 100)),
              gpu_peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20), gpu=torch.cuda.get_device_name(0),
              det_model=Path(a.model).name, mark_model=str(a.mark_model))
    if a.gt: st["eval"] = evaluate_gt(ev, a.gt, dur, out, t0=float(pts[start_f] - pts[0]))
    json.dump(st, open(out / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_report(out / "report.html", st, ev)
    print(json.dumps(st, ensure_ascii=False, indent=1))


def evaluate_gt(ev, gt_path, dur, out, tol=2.0, t0=0.0):
    g = pd.read_csv(gt_path, dtype={"label": str}, keep_default_na=False, encoding="utf-8-sig")
    g = g[(g.note == "") & (g.label != "invalid") & (g.time_s.astype(float) >= t0 - 1) & (g.time_s.astype(float) <= t0 + dur)].copy(); g["time_s"] = g.time_s.astype(float)
    g["gdec"] = np.where(g.label == "cargo", "cargo", "exclude")
    used = set(); match = []
    for r in ev.itertuples():
        c = g[(g.dir == r.dir) & (~g.index.isin(used)) & ((g.time_s - r.cross_s).abs() <= tol)]
        if len(c):
            j = (c.time_s - r.cross_s).abs().idxmin(); used.add(j); match.append((r.event_no, j))
    m = {e: j for e, j in match}
    ev = ev.assign(gt_id=[g.id[m[e]] if e in m else "" for e in ev.event_no], gt_label=[g.label[m[e]] if e in m else "" for e in ev.event_no])
    ev.to_csv(out / "events_eval.csv", index=False, encoding="utf-8-sig")
    miss = g[~g.index.isin(used)]; miss.to_csv(out / "missed_gt.csv", index=False, encoding="utf-8-sig")
    res = {}
    for op in ("load", "unload"):
        gc = int(((g.op == op) & (g.gdec == "cargo")).sum()); pc = int(((ev.op == op) & (ev.decision == "cargo")).sum())
        res[op] = dict(gt_cargo=gc, pred_cargo=pc, diff=pc - gc)
    mm = ev[ev.gt_id != ""]; gl = mm.gt_label.map(lambda l: "cargo" if l == "cargo" else "exclude")
    res.update(gt_events=len(g), pred_events=len(ev), matched=len(mm), false_events=int((ev.gt_id == "").sum()), missed_events=len(miss),
               decision_wrong=int((gl != mm.decision).sum()), cargo_as_exclude=int(((gl == "cargo") & (mm.decision == "exclude")).sum()),
               exclude_as_cargo=int(((gl == "exclude") & (mm.decision == "cargo")).sum()), tol_s=tol)
    return res


def write_report(path, st, ev):
    c = st["counts"]; dur = max(st["duration_s"], 1)
    def f(x): return "-" if x is None else f"{x*1000:.0f} ms" if x < 1 else f"{x:.1f} s"
    pts_svg = []
    for r in ev.itertuples():
        x = 40 + 900 * r.cross_s / dur; y = 40 if r.op == "load" else 100
        col = "var(--cargo)" if r.decision == "cargo" else "var(--excl)"
        pts_svg.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="{col}"><title>#{r.event_no} {r.cross_time} {r.op} {r.decision} {r.reason}</title></circle>')
    ticks = "".join(f'<line x1="{40+900*t/dur:.1f}" y1="20" x2="{40+900*t/dur:.1f}" y2="125" class="grid"/><text x="{40+900*t/dur:.1f}" y="140" class="tk">{int(t//60)}분</text>'
                    for t in np.arange(0, dur + 1, max(60, round(dur / 8 / 60) * 60)))
    trs = "".join(f"<tr><td>{r.event_no}</td><td>{r.cross_time}</td><td>{'선적' if r.op=='load' else '하역'}</td>"
                  f"<td><span class='pill {r.decision}'>{'화물' if r.decision=='cargo' else '제외'}</span></td><td>{r.reason}</td><td>{r.latency_s:.1f}s</td></tr>" for r in ev.itertuples())
    evl = st.get("eval")
    evh = "" if not evl else f"""<section><h2>정답 대조 (참고)</h2><div class="kv">
      <div><b>{evl['matched']}/{evl['gt_events']}</b><span>매칭 이벤트 (±{evl['tol_s']}s, 같은 방향)</span></div>
      <div><b>{evl['missed_events']}</b><span>미탐</span></div><div><b>{evl['false_events']}</b><span>오탐 이벤트</span></div>
      <div><b>{evl['decision_wrong']}</b><span>판정 오류 (화물→제외 {evl['cargo_as_exclude']}, 제외→화물 {evl['exclude_as_cargo']})</span></div>
      <div><b>{evl['load']['pred_cargo']} / {evl['load']['gt_cargo']}</b><span>선적 화물 예측/정답</span></div>
      <div><b>{evl['unload']['pred_cargo']} / {evl['unload']['gt_cargo']}</b><span>하역 화물 예측/정답</span></div></div></section>"""
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>실시간 카운트 리포트</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#17212b;--mut:#5b6672;--line:#e3e6ea;--cargo:#1f6fd1;--excl:#c2541b;--acc:#147a4b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#11161c;--card:#1a2129;--fg:#e6ebf0;--mut:#98a3ae;--line:#2a333d;--cargo:#6aa8f5;--excl:#f0935c;--acc:#55cf94}}}}
:root[data-theme="dark"]{{--bg:#11161c;--card:#1a2129;--fg:#e6ebf0;--mut:#98a3ae;--line:#2a333d;--cargo:#6aa8f5;--excl:#f0935c;--acc:#55cf94}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Malgun Gothic",sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 48px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mut);margin:0 0 20px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:0 0 16px}}h2{{font-size:16px;margin:0 0 12px}}
.big{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}.big div{{border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.big b{{font-size:30px;display:block;font-variant-numeric:tabular-nums}}.big span,.kv span{{color:var(--mut);font-size:13px}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}.kv b{{display:block;font-size:18px;font-variant-numeric:tabular-nums}}
svg{{width:100%;height:auto}}.grid{{stroke:var(--line)}}.tk,.lb{{fill:var(--mut);font-size:11px}}.lb{{font-size:12px}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--mut);font-weight:600}}.pill{{padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}}.pill.cargo{{background:var(--cargo)}}.pill.exclude{{background:var(--excl)}}
.legend{{color:var(--mut);font-size:13px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px}}
</style></head><body><main>
<h1>실시간 카운트 리포트</h1><p class="sub">{st['video']} · 실시간 RTSP 모의 ({'모든 프레임 처리' if st['mode']=='no-drop' else '최신 프레임만 처리, 밀리면 버림'}) · 영상 {st['duration_s']/60:.1f}분</p>
<section><h2>대수</h2><div class="big">
<div><b>{c['load_cargo']}</b><span>선적 화물차 (야드→배)</span></div><div><b>{c['unload_cargo']}</b><span>하역 화물차 (배→야드)</span></div>
<div><b>{c['load_exclude']+c['unload_exclude']}</b><span>제외 (세이프티·작업) · 선적 {c['load_exclude']} / 하역 {c['unload_exclude']}</span></div></div></section>
<section><h2>타임라인</h2><p class="legend"><span class="dot" style="background:var(--cargo)"></span>화물<span class="dot" style="background:var(--excl)"></span>제외 · 점에 마우스를 올리면 상세</p>
<svg viewBox="0 0 960 150" role="img" aria-label="통과 시각 타임라인">{ticks}<text x="0" y="44" class="lb">선적</text><text x="0" y="104" class="lb">하역</text>{''.join(pts_svg)}</svg></section>
<section><h2>처리 성능</h2><div class="kv">
<div><b>{st['realtime_factor']:.2f}×</b><span>실시간 배율 (영상 길이 / 처리 시간)</span></div>
<div><b>{st['processed_fps']} / {st['fps_source']:.1f}</b><span>처리 fps / 카메라 fps</span></div>
<div><b>{st['frames_dropped']:,}</b><span>버린 프레임 (받은 {st['frames_received']:,})</span></div>
<div><b>{f(st['det_latency_s']['p50'])} · {f(st['det_latency_s']['p95'])}</b><span>프레임당 탐지+추적 p50 · p95</span></div>
<div><b>{f(st['mark_latency_s']['p50'])} · {f(st['mark_latency_s']['p95'])}</b><span>차량당 표식 판정 p50 · p95</span></div>
<div><b>{f((st['decision_latency_s'] or {}).get('p50'))} · {f((st['decision_latency_s'] or {}).get('p95'))}</b><span>선 통과 → 판정 확정 p50 · p95</span></div>
<div><b>{st['decisions_lt_min_crops']}건</b><span>크롭 {st['min_crops']}장 미만으로 판정 (판정 시점: {'선 통과 즉시' if st['decide_at']=='cross' else '체인 확정 후'})</span></div>
<div><b>{len(st['replay']['segments'])}구간</b><span>스트림 재전송 감지 (건너뛴 프레임 {st['replay']['frames_skipped']})</span></div>
<div><b>{st['gpu_peak_alloc_mb']:,} MB</b><span>GPU 메모리 최대 (PyTorch 할당)</span></div>
<div><b>{st['wall_s']/60:.1f}분</b><span>총 처리 시간</span></div></div></section>
{evh}
<section><h2>이벤트 {len(ev)}건</h2><div class="tbl"><table><thead><tr><th>#</th><th>통과 시각</th><th>방향</th><th>판정</th><th>근거</th><th>판정 지연</th></tr></thead><tbody>{trs}</tbody></table></div></section>
</main></body></html>"""
    Path(path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
