"""COUNT 현장 실행: RTSP 카메라(또는 영상 파일)를 받아 기준선을 넘는 화물차를 선적·하역별로 센다.

  python app/count_live.py                      # config.json 의 source 로 실행 (시작 시 기준선 확인 화면)
  python app/count_live.py --setup              # 기준선을 새로 긋고 시작
  python app/count_live.py --source test.mp4    # 영상 파일로 시험 (실시간 속도로 재생)
  python app/count_live.py --no-show            # 화면 없이 (기준선이 config 에 저장돼 있어야 함)

출력 (logs/):
  count.db                    SQLite DB — runs(실행) · events(차량 1대 1행) · daily_counts(일별 집계 뷰)
  media/YYYYMMDD/             차량별 통과 순간 스크린샷(_shot.jpg) · 통과 전후 클립(_clip.mp4) · 판정 크롭(_crop.jpg)
  events_YYYYMMDD.csv         DB 와 같은 내용의 사람용 사본 (엑셀)
  status.json                 오늘 누적 대수·처리 상태 (5초마다 갱신)
  run_YYYYMMDD.log            실행 로그
"""
import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # RTSP 는 TCP 로 (UDP 패킷 손실로 화면 깨짐 방지)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent   # app/
ROOT = HERE.parent                       # 패키지 루트 (config.json, models/, logs/)
sys.path.insert(0, str(HERE))
from core import Counter  # noqa: E402
from line_setup import setup_line, _font  # noqa: E402
from storage import EventDB, FrameBuffer, MediaWriter  # noqa: E402

APP_VERSION = "1.2-rc2"

OP_KO = {"load": "선적", "unload": "하역"}
DEC_KO = {"cargo": "화물", "exclude": "제외"}


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        d = ROOT / "logs"; d.mkdir(exist_ok=True)
        with open(d / f"run_{dt.date.today():%Y%m%d}.log", "a", encoding="utf-8") as f: f.write(line + "\n")
    except OSError: pass


# ---------------------------------------------------------------- 입력 스트림
def next_fi(last_fi, reconnected, now, t_last, fps):
    """RTSP 프레임 번호: 디코딩한 프레임마다 +1. 재접속 직후 첫 프레임만 끊겨 있던 시간 × fps 만큼 건너뛴다."""
    if reconnected and t_last is not None:
        return last_fi + max(1, int(round((now - t_last) * fps)))
    return last_fi + 1


class Stream(threading.Thread):
    """최신 프레임 1장만 보관(처리가 밀리면 이전 프레임은 버림). RTSP 가 끊기면 자동 재접속.
    fi 는 카메라 시간축의 프레임 번호 = 디코딩한 프레임 수 (처리 못 해 버린 프레임 포함).
    도착 시각으로 계산하지 않는다: TCP RTSP 는 멈췄다가 몰아서 들어오는 일이 흔해, 도착 시각 기준이면
    실제 1프레임 차이를 여러 프레임 건너뛴 것으로 착각해 칼만 예측이 과하게 앞서 나간다.
    재접속으로 끊긴 동안만 경과 시간 × fps 만큼 번호를 건너뛴다."""
    def __init__(self, src, fps_cfg, file_realtime=True):
        super().__init__(daemon=True)
        self.src = int(src) if str(src).isdigit() else src
        self.is_file = isinstance(self.src, str) and Path(self.src).exists()
        self.fps_cfg = fps_cfg; self.file_realtime = file_realtime
        self.lock = threading.Lock(); self.ev = threading.Event(); self.latest = None
        self.fps = None; self.size = None; self.received = 0; self.dropped = 0; self.reconnects = 0
        self.gap = False; self.done = False; self.stop = False; self.connected = threading.Event()
        self.go = threading.Event()  # 파일 입력: 기준선 설정이 끝날 때까지 첫 프레임에서 멈춤
        self.wall_base = None        # 프레임 번호 0 의 벽시계 시각 (도착 기록이 없을 때 어림용)
        self.arr = {}                # fi → 실제 도착 벽시계 시각 (최근 것만)
        self.buf = None              # storage.FrameBuffer (스크린샷·클립용 최근 프레임)

    def _open(self):
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG) if isinstance(self.src, str) else cv2.VideoCapture(self.src)
        if not cap.isOpened(): return None
        f = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(self.fps_cfg) if self.fps_cfg else (f if 1 <= f <= 120 else 15.0)
        return cap

    def run(self):
        last_fi = -1; first = True; t_last = None
        while not self.stop:
            cap = self._open()
            if cap is None:
                if self.is_file: log(f"영상 파일을 열 수 없음: {self.src}"); break
                log("카메라 연결 실패, 5초 뒤 재시도"); time.sleep(5); continue
            if not first: self.reconnects += 1; self.gap = True; log(f"카메라 재접속 ({self.reconnects}회)")
            jump = not first; first = False; fail = 0; tf0 = time.perf_counter()
            while not self.stop:
                ok, img = cap.read()
                if not ok:
                    fail += 1
                    if self.is_file or fail > 25: break
                    time.sleep(0.04); continue
                fail = 0; now = time.perf_counter()
                if self.size is None: self.size = (img.shape[1], img.shape[0])
                if self.is_file:
                    fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if self.received == 1:  # 첫 프레임 넘긴 뒤 설정 화면이 끝날 때까지 대기
                        self.go.wait(); tf0 = time.perf_counter() - cap.get(cv2.CAP_PROP_POS_MSEC) / 1000; now = time.perf_counter()
                        self.wall_base = time.time() - cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if self.file_realtime:  # 파일도 카메라처럼 제 속도로 흘려보냄
                        due = tf0 + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        if due > now: time.sleep(due - now)
                else:
                    if self.wall_base is None: self.wall_base = time.time()
                    fi = next_fi(last_fi, jump, now, t_last, self.fps); jump = False
                t_last = now
                fi = max(fi, last_fi + 1); last_fi = fi
                self.arr[fi] = time.time()
                if len(self.arr) > 4000:
                    for k in list(self.arr)[:2000]: del self.arr[k]
                if self.buf is not None: self.buf.put(fi, img)
                with self.lock:
                    if self.latest is not None: self.dropped += 1
                    self.latest = (fi, img)
                self.received += 1; self.connected.set(); self.ev.set()
            cap.release()
            if self.is_file: break
            log("카메라 스트림 끊김")
        self.done = True; self.ev.set()

    def wall_of(self, fi):
        """프레임 번호 → 카메라에서 도착한 시각. 통과 프레임은 관측 프레임 사이 보간값이라 가까운 앞뒤 도착 시각으로 보간."""
        arr = dict(list(self.arr.items()))  # 수신 스레드가 지우는 중일 수 있어 사본으로 (KeyError 방지)
        ks = [k for k in arr if abs(k - fi) <= 3 * self.fps]
        lo = max((k for k in ks if k <= fi), default=None); hi = min((k for k in ks if k >= fi), default=None)
        if lo is not None and hi is not None:
            return arr[lo] if hi == lo else arr[lo] + (arr[hi] - arr[lo]) * (fi - lo) / (hi - lo)
        if lo is not None: return arr[lo] + (fi - lo) / self.fps
        if hi is not None: return arr[hi] - (hi - fi) / self.fps
        base = self.wall_base if self.wall_base is not None else time.time()
        return base + fi / self.fps

    def get(self, timeout=1.0):
        """(fi, img) 또는 None(대기 시간 초과). 스트림이 끝났으면 StopIteration."""
        if not self.ev.wait(timeout):
            return None
        with self.lock:
            x = self.latest; self.latest = None; self.ev.clear()
        if x is None and self.done: raise StopIteration
        return x


# ---------------------------------------------------------------- 화면
class View:
    COL = {"cargo": (90, 200, 40), "exclude": (50, 60, 235), None: (180, 175, 170)}  # BGR

    def __init__(self, W, H, width, line):
        self.s = min(1.0, width / W); self.size = (int(W * self.s), int(H * self.s))
        self.A = tuple(int(v * self.s) for v in line[0]); self.B = tuple(int(v * self.s) for v in line[1])
        self.f_big, self.f_sm = _font(30), _font(17); self.hud = None; self.key = None

    def _hud(self, c, fps, drop):
        from PIL import Image, ImageDraw
        key = (c["load_cargo"], c["unload_cargo"], c["exclude"], round(fps), round(drop, 1))
        if key == self.key: return self.hud
        im = Image.new("RGB", (400, 168), (26, 20, 15)); d = ImageDraw.Draw(im)
        for i, (name, v, col) in enumerate((("선적 화물 (오늘)", c["load_cargo"], (255, 200, 120)),
                                            ("하역 화물 (오늘)", c["unload_cargo"], (255, 200, 120)),
                                            ("제외 (세이프티·작업)", c["exclude"], (120, 150, 255)))):
            y = 10 + i * 40
            d.text((14, y + 6), name, font=self.f_sm, fill=(232, 226, 220)); d.text((386, y), str(v), font=self.f_big, anchor="ra", fill=col)
        d.text((14, 138), f"처리 {fps:.1f} fps · 버린 프레임 {drop:.1f}%   [Q: 종료]", font=self.f_sm, fill=(170, 170, 170))
        self.hud = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR); self.key = key
        return self.hud

    def show(self, img, vis, counts, fps, drop):
        v = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        cv2.line(v, self.A, self.B, (255, 230, 0), 2, cv2.LINE_AA)
        for tid, (x1, y1, x2, y2), ev in vis:
            dec = ev.get("decision") if ev else None
            col = self.COL[dec]; p1 = (int(x1 * self.s), int(y1 * self.s)); p2 = (int(x2 * self.s), int(y2 * self.s))
            cv2.rectangle(v, p1, p2, col, 2)
            lab = f"#{ev.get('id', '')} {'CARGO' if dec == 'cargo' else 'EXCL' if dec == 'exclude' else '...'}" if ev else f"{tid}"
            cv2.putText(v, lab, (p1[0] + 3, max(14, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        h = self._hud(counts, fps, drop); hh, hw = h.shape[:2]
        H_, W_ = v.shape[:2]; roi = v[H_ - hh - 10:H_ - 10, W_ - hw - 10:W_ - 10]  # 오른쪽 아래 (위쪽은 카메라 시각 표시); cv2.addWeighted(h, 0.85, roi, 0.15, 0, roi)
        cv2.imshow("COUNT", v)
        k = cv2.waitKey(1) & 0xFF
        return k in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("COUNT", cv2.WND_PROP_VISIBLE) < 1


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="COUNT 현장 실행")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--source", default=None, help="RTSP 주소 / 영상 파일 / 웹캠 번호 (config 보다 우선)")
    ap.add_argument("--setup", action="store_true", help="기준선을 새로 긋고 시작")
    ap.add_argument("--no-show", action="store_true", help="화면 없이 실행")
    ap.add_argument("--test", action="store_true", help="시험 실행: 기준선을 새로 긋되 config 에 저장하지 않고, 기록은 logs_test/ 에")
    ap.add_argument("--max-seconds", type=float, default=0, help="이 시간(초)만 실행하고 종료 (시험용)")
    a = ap.parse_args()

    cfg_path = Path(a.config); cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if a.test: a.setup = True; cfg["log_dir"] = "logs_test"
    for k in ("vehicle", "marks"):
        p = Path(cfg["models"][k]); cfg["models"][k] = str(p if p.is_absolute() else ROOT / p)
    src = a.source or cfg["source"]; show = cfg.get("show_window", True) and not a.no_show
    log(f"시작: 입력 {src if not str(src).startswith('rtsp') else src.split('@')[-1]}")

    stream = Stream(src, cfg.get("fps"), cfg.get("file_realtime", True)); stream.start()
    if not stream.connected.wait(60): log("60초 안에 영상이 들어오지 않아 종료"); return 1
    first = stream.get(5)
    while first is None and not stream.done: first = stream.get(5)
    if first is None: log("첫 프레임을 받지 못해 종료"); return 1
    W, H = stream.size; fps = stream.fps
    log(f"영상 {W}x{H} @ {fps:.1f}fps")

    # 기준선: 시작할 때 현재 화각에서 확인/설정
    saved = cfg.get("line")
    if saved and saved.get("frame_size") and tuple(saved["frame_size"]) != (W, H):  # 해상도 바뀜 → 비율 환산
        sx, sy = W / saved["frame_size"][0], H / saved["frame_size"][1]
        saved = dict(saved, line=[[p[0] * sx, p[1] * sy] for p in saved["line"]], frame_size=[W, H])
    if show and (a.setup or not saved or cfg.get("confirm_line_on_start", True)):
        res = setup_line(first[1], None if a.setup else saved, cfg.get("line_confirm_timeout_s", 15))
        if res is None: log("기준선 설정 화면에서 종료"); stream.stop = True; stream.go.set(); return 0
        if res != saved and not a.test:
            cfg_disk = json.loads(cfg_path.read_text(encoding="utf-8")); cfg_disk["line"] = res
            cfg_path.write_text(json.dumps(cfg_disk, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"기준선 저장: {res['line']} (야드 쪽 {res['yard_side']:+d})")
        saved = res
    if not saved: log("기준선이 없습니다. 화면을 켜고(--setup) 먼저 기준선을 그어 주세요."); stream.stop = True; stream.go.set(); return 1

    log_dir = Path(cfg.get("log_dir", "logs")); log_dir = log_dir if log_dir.is_absolute() else ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True); cfg["log_dir_abs"] = str(log_dir)
    cam = cfg.get("camera_id", "cam1")
    db = EventDB(log_dir / "count.db")
    src_safe = str(src) if not str(src).startswith("rtsp") else "rtsp://" + str(src).split("@")[-1]
    run_id = db.start_run(camera_id=cam, source=src_safe, frame_w=W, frame_h=H, fps=round(fps, 3), line_json=json.dumps(saved),
                          vehicle_model=Path(cfg["models"]["vehicle"]).name, marks_model=Path(cfg["models"]["marks"]).name, app_version=APP_VERSION)
    buf = FrameBuffer(W, cfg.get("media_width", 1280), cfg.get("clip_pre_s", 1.5) + cfg.get("clip_post_s", 1.5) + 6, fps)
    stream.buf = buf if (cfg.get("save_screenshots", True) or cfg.get("save_clips", True)) else None
    media = MediaWriter(cfg, db, buf, run_id, cam, fps, saved["line"], stream.wall_of, log, _font(22)); media.start()

    def on_decision(ev): media.q.put(ev)

    cfg["_log"] = log
    log("모델 불러오는 중...")
    cnt = Counter(cfg, W, H, fps, saved["line"], saved["yard_side"], on_decision=on_decision)
    log(f"실행 중 (장치: {'GPU' if cnt.device == 0 else 'CPU — 실시간 처리가 어려울 수 있음'}, "
        f"차량 탐지 {'TensorRT' if cnt.det_trt else 'PyTorch'} {cnt.imgsz[0]}x{cnt.imgsz[1]}, 표식 {'TensorRT' if cnt.mark_trt else 'PyTorch'})")
    view = View(W, H, cfg.get("display_width", 1280), saved["line"]) if show else None
    stream.go.set()

    t_run = time.time()
    t_stat = t_fps = time.time(); n_fps = 0; cur_fps = 0.0; drop0 = (0, 0)
    try:
        while True:
            try: x = stream.get(1.0)
            except StopIteration: break
            if stream.gap: stream.gap = False; cnt.reset_tracks()
            if x is None: continue
            fi, img = x
            if a.max_seconds and (fi - first[0]) / fps > a.max_seconds: break
            vis = cnt.process(fi, img); n_fps += 1
            now = time.time()
            if now - t_fps >= 2: cur_fps = n_fps / (now - t_fps); n_fps = 0; t_fps = now
            rc, dr = stream.received - drop0[0], stream.dropped - drop0[1]
            drop = 100 * dr / max(1, rc)
            if now - t_stat >= 5:
                t_stat = now; drop0 = (stream.received, stream.dropped)
                media.status(dict(running=True, fps_camera=round(fps, 2), fps_processed=round(cur_fps, 2), drop_pct=round(drop, 1),
                                reconnects=stream.reconnects, device="GPU" if cnt.device == 0 else "CPU"))
            if view and view.show(img, vis, media.today, cur_fps, drop): break
    except KeyboardInterrupt:
        pass
    finally:
        log("종료 중 (진행 중인 판정 마무리)")
        cnt.close()                   # 남은 표식 판정을 끝까지 (→ media 큐로 들어감)
        stream.stop = True
        media.close()                 # 큐·클립을 다 쓴 뒤 종료
        media.status(dict(running=False)); db.end_run(run_id)
        if view: cv2.destroyAllWindows()
        el = max(1e-6, time.time() - t_run)
        log(f"처리 {cnt.proc / el:.1f} fps (카메라 {fps:.1f}) · 받은 프레임 {stream.received} · 버린 프레임 {stream.dropped} ({100 * stream.dropped / max(1, stream.received):.1f}%)")
        log(f"종료. 이번 실행: 선적 화물 {media.session['load_cargo']} / 하역 화물 {media.session['unload_cargo']} / 제외 {media.session['exclude']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
