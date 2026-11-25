"""기록 저장: 로컬 SQLite DB + 차량별 미디어(통과 순간 스크린샷 · 전후 클립 · 판정 크롭) + 날짜별 CSV.

DB 는 표준 SQL 컬럼만 쓰므로 나중에 PostgreSQL/MySQL 등 중앙 DB 로 옮길 때 같은 스키마를 그대로 만들고
synced=0 인 행만 올리면 된다 (README 의 'DB 연동' 참고). 미디어는 파일로 두고 DB 에는 상대 경로만 넣는다.
"""
import csv
import datetime as dt
import json
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,          -- 실행 1회 (프로그램 시작~종료)
    camera_id     TEXT NOT NULL,
    started_at    TEXT NOT NULL,             -- 'YYYY-MM-DD HH:MM:SS' 현지 시각
    ended_at      TEXT,
    source        TEXT,                      -- 입력 (RTSP 계정 정보는 지움)
    frame_w       INTEGER, frame_h INTEGER, fps REAL,
    line_json     TEXT,                      -- 기준선·야드 쪽
    vehicle_model TEXT, marks_model TEXT,
    app_version   TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,   -- 차량 번호 (전체 기간 고유)
    camera_id       TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    crossed_at      TEXT NOT NULL,           -- 기준선 통과 시각 'YYYY-MM-DD HH:MM:SS.mmm' 현지
    crossed_ts      REAL NOT NULL,           -- 같은 시각 UNIX epoch(초) — 시간대 무관 정렬·연동용
    direction       TEXT NOT NULL CHECK (direction IN ('load','unload')),     -- 선적(야드→배) / 하역(배→야드)
    vehicle_class   TEXT NOT NULL CHECK (vehicle_class IN ('cargo','exclude')), -- 화물 / 제외(세이프티·작업차)
    marks_plate     INTEGER NOT NULL DEFAULT 0,  -- 표식이 보인 크롭 수
    marks_beacon    INTEGER NOT NULL DEFAULT 0,
    marks_number    INTEGER NOT NULL DEFAULT 0,
    n_crops         INTEGER,
    reason          TEXT,
    decided_at      TEXT,                    -- 판정 확정 시각
    latency_s       REAL,                    -- 통과 → 판정 (초)
    screenshot_path TEXT,                    -- 미디어: log_dir 기준 상대 경로
    clip_path       TEXT,
    crop_path       TEXT,
    synced          INTEGER NOT NULL DEFAULT 0,  -- 중앙 DB 업로드 여부 (연동 시 사용)
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(camera_id, crossed_ts);
CREATE INDEX IF NOT EXISTS idx_events_sync ON events(synced);
CREATE VIEW IF NOT EXISTS daily_counts AS
    SELECT camera_id, substr(crossed_at, 1, 10) AS day,
           SUM(direction = 'load'   AND vehicle_class = 'cargo') AS load_cargo,
           SUM(direction = 'unload' AND vehicle_class = 'cargo') AS unload_cargo,
           SUM(vehicle_class = 'exclude')                        AS excluded,
           COUNT(*)                                              AS total
    FROM events GROUP BY camera_id, day;
"""

OP_KO = {"load": "선적", "unload": "하역"}
DEC_KO = {"cargo": "화물", "exclude": "제외"}


def now_str(t=None, ms=False):
    d = dt.datetime.fromtimestamp(t if t is not None else time.time())
    return f"{d:%Y-%m-%d %H:%M:%S.%f}"[:-3] if ms else f"{d:%Y-%m-%d %H:%M:%S}"


class EventDB:
    def __init__(self, path):
        self.c = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.c.execute("PRAGMA journal_mode=WAL"); self.c.executescript(SCHEMA); self.c.commit()
        self.lock = threading.Lock()

    def exec(self, sql, args=()):
        with self.lock:
            cur = self.c.execute(sql, args); self.c.commit(); return cur

    def start_run(self, **kw):
        run_id = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        kw = dict(run_id=run_id, started_at=now_str(), **kw)
        self.exec(f"INSERT INTO runs ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values()))
        return run_id

    def end_run(self, run_id): self.exec("UPDATE runs SET ended_at=? WHERE run_id=?", (now_str(), run_id))

    def insert_event(self, **kw):
        kw["created_at"] = now_str()
        return self.exec(f"INSERT INTO events ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values())).lastrowid

    def update(self, eid, **kw):
        self.exec(f"UPDATE events SET {','.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), eid))

    def counts(self, camera_id, day):
        r = self.exec("SELECT load_cargo, unload_cargo, excluded FROM daily_counts WHERE camera_id=? AND day=?", (camera_id, day)).fetchone()
        return dict(load_cargo=r[0] or 0, unload_cargo=r[1] or 0, exclude=r[2] or 0) if r else dict(load_cargo=0, unload_cargo=0, exclude=0)


class FrameBuffer:
    """받은 프레임을 모두(처리 못 하고 버린 프레임 포함) 축소 JPEG 로 몇 초간 보관 → 스크린샷·클립 원본.
    put() 은 큐에 넣기만 하고 변환은 전용 스레드가 한다 (프레임 수신·판정 스레드를 붙잡지 않도록).
    변환이 밀리면 클립 프레임을 건너뛴다 — 판정용 프레임은 영향받지 않는다."""
    def __init__(self, W, width, keep_s, fps, quality=85):
        self.s = min(1.0, width / W); self.keep = int(keep_s * fps) + 10; self.q = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.buf = OrderedDict(); self.lock = threading.Lock(); self.last = -1; self.skipped = 0
        self.inq = queue.Queue(maxsize=int(fps)); threading.Thread(target=self._loop, daemon=True).start()

    def put(self, fi, img):
        try: self.inq.put_nowait((fi, img))
        except queue.Full: self.skipped += 1
        self.last_in = fi

    def _loop(self):
        while True:
            fi, img = self.inq.get()
            small = cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA) if self.s < 1 else img
            jpg = cv2.imencode(".jpg", small, self.q)[1]
            with self.lock:
                self.buf[fi] = jpg; self.last = fi
                while len(self.buf) > self.keep: self.buf.popitem(last=False)

    def range(self, f0, f1):
        with self.lock: return [(k, v) for k, v in self.buf.items() if f0 <= k <= f1]

    def nearest(self, fi):
        with self.lock:
            if not self.buf: return None, None
            k = min(self.buf, key=lambda x: abs(x - fi)); return k, self.buf[k]


def _ffmpeg():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class MediaWriter(threading.Thread):
    """판정이 난 차량마다: DB 기록 → 통과 순간 스크린샷·판정 크롭 저장 → CSV → (통과 + post 초 영상이 들어오면) 클립 저장."""
    def __init__(self, cfg, db, buf, run_id, camera_id, fps, line, wall_of, log, font):
        super().__init__(daemon=True)
        self.root = Path(cfg["log_dir_abs"]); self.db = db; self.buf = buf; self.run_id = run_id; self.cam = camera_id
        self.fps = fps; self.wall_of = wall_of; self.log = log; self.font = font
        self.pre, self.post = cfg.get("clip_pre_s", 1.5), cfg.get("clip_post_s", 1.5)
        self.save_shot, self.save_clip, self.save_crop = cfg.get("save_screenshots", True), cfg.get("save_clips", True), cfg.get("save_crops", True)
        s = buf.s; self.A = tuple(int(v * s) for v in line[0]); self.B = tuple(int(v * s) for v in line[1])
        self.q = queue.Queue(); self.pending = []; self.stop_flag = False; self.ffmpeg = _ffmpeg()
        self.lock = threading.Lock(); self.day = None; self.today = None; self.session = dict(load_cargo=0, unload_cargo=0, exclude=0); self.last = None
        self._roll()

    # ---- 오늘 누적 (DB 기준, 재시작해도 이어짐)
    def _roll(self):
        """자정이 지나면 '오늘' 누적을 DB 에서 새 날짜로 다시 읽는다 (status 갱신 때도 호출)."""
        d = dt.date.today()
        if d != self.day: self.day = d; self.today = self.db.counts(self.cam, f"{d}")

    @staticmethod
    def _add(c, ev):
        k = "exclude" if ev["decision"] == "exclude" else f"{ev['op']}_cargo"; c[k] += 1

    def run(self):
        while not (self.stop_flag and self.q.empty() and not self.pending):
            try: ev = self.q.get(timeout=0.2)
            except queue.Empty: ev = None
            if ev is not None:
                try: self._record(ev)
                except Exception as e: self.log(f"기록 오류 (#{ev.get('no')}): {e!r}")
            for job in list(self.pending):  # 클립: 통과 + post 초까지 프레임이 들어왔거나 종료 중이면 저장
                if self.buf.last >= job["f1"] or self.stop_flag or time.time() - job["t"] > self.post + 10:
                    self.pending.remove(job)
                    try: self._clip(job)
                    except Exception as e: self.log(f"클립 저장 오류 (id {job['id']}): {e!r}")

    def _record(self, ev):
        tc = self.wall_of(ev["frame"]); m = ev.get("marks") or {}
        with self.lock:
            self._roll()
            if dt.date.fromtimestamp(tc) == self.day: self._add(self.today, ev)  # 날짜는 기준선 통과 시각 기준 (DB 집계와 같게)
            self._add(self.session, ev)
        eid = self.db.insert_event(camera_id=self.cam, run_id=self.run_id, crossed_at=now_str(tc, ms=True), crossed_ts=round(tc, 3),
                                   direction=ev["op"], vehicle_class=ev["decision"], marks_plate=m.get("plate", 0),
                                   marks_beacon=m.get("beacon", 0), marks_number=m.get("number", 0), n_crops=ev["n_crops"],
                                   reason=ev["reason"], decided_at=now_str(ev["decided_wall"], ms=True), latency_s=round(ev["decided_wall"] - tc, 2))
        ev["id"] = eid
        d = dt.datetime.fromtimestamp(tc); sub = Path("media") / f"{d:%Y%m%d}"; (self.root / sub).mkdir(parents=True, exist_ok=True)
        stem = f"{eid:06d}_{d:%H%M%S}_{ev['op']}_{ev['decision']}"
        label = f"#{eid}  {d:%H:%M:%S}  {OP_KO[ev['op']]} {DEC_KO[ev['decision']]}"
        boxes = (ev.get("boxes") or {}).copy()
        paths = {}
        if self.save_shot:
            k, jpg = self.buf.nearest(ev["frame"])
            if jpg is not None:
                img = self._draw(cv2.imdecode(jpg, cv2.IMREAD_COLOR), boxes, k, ev, label)
                cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(self.root / sub / f"{stem}_shot.jpg"))
                paths["screenshot_path"] = str(sub / f"{stem}_shot.jpg")
        if self.save_crop and ev.get("snapshot") is not None:
            cv2.imencode(".jpg", ev["snapshot"], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(self.root / sub / f"{stem}_crop.jpg"))
            paths["crop_path"] = str(sub / f"{stem}_crop.jpg")
        if paths: self.db.update(eid, **paths)
        ev["snapshot"] = None  # 메모리 정리
        self._csv(eid, d, ev, paths)
        with self.lock: self.last = dict(id=eid, time=f"{d:%H:%M:%S}", op=OP_KO[ev["op"]], decision=DEC_KO[ev["decision"]])
        self.log(f"#{eid} {d:%H:%M:%S} {OP_KO[ev['op']]} {DEC_KO[ev['decision']]}  ({ev['reason']})  "
                 f"오늘: 선적 {self.today['load_cargo']} / 하역 {self.today['unload_cargo']} / 제외 {self.today['exclude']}")
        if self.save_clip:
            self.pending.append(dict(id=eid, ev=ev, boxes=boxes, label=label, sub=sub, stem=stem, t=time.time(),
                                     f0=ev["frame"] - int(self.pre * self.fps), f1=ev["frame"] + int(self.post * self.fps)))

    def _csv(self, eid, d, ev, paths):
        p = self.root / f"events_{d:%Y%m%d}.csv"; new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new: w.writerow(["번호", "통과 시각", "방향", "판정", "근거", "판정 지연(초)", "스크린샷", "클립"])
            w.writerow([eid, f"{d:%Y-%m-%d %H:%M:%S.%f}"[:-5], OP_KO[ev["op"]], DEC_KO[ev["decision"]], ev["reason"],
                        round(ev["decided_wall"] - d.timestamp(), 2), paths.get("screenshot_path", ""),
                        str(paths.get("screenshot_path", "")).replace("_shot.jpg", "_clip.mp4") if self.save_clip else ""])

    def _draw(self, img, boxes, fi, ev, label):
        s = self.buf.s
        cv2.line(img, self.A, self.B, (255, 230, 0), 2, cv2.LINE_AA)
        if boxes:
            k = min(boxes, key=lambda x: abs(x - fi))
            if abs(k - fi) <= max(3, self.fps * 0.3):
                x1, y1, x2, y2 = (int(v * s) for v in boxes[k])
                col = (90, 200, 40) if ev["decision"] == "cargo" else (50, 60, 235)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        from PIL import Image, ImageDraw
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im, "RGBA")
        tw = d.textlength(label, font=self.font); H = im.size[1]
        d.rounded_rectangle([10, H - 50, tw + 34, H - 10], 8, fill=(15, 20, 26, 210)); d.text((22, H - 46), label, font=self.font, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def _clip(self, job):
        frames = self.buf.range(job["f0"], job["f1"])
        if len(frames) < 2: return
        out = self.root / job["sub"] / f"{job['stem']}_clip.mp4"
        first = cv2.imdecode(frames[0][1], cv2.IMREAD_COLOR); h, w = first.shape[:2]
        if self.ffmpeg:
            p = subprocess.Popen([self.ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps:.3f}",
                                  "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                                 stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            put = lambda im: p.stdin.write(im.tobytes())
        else:  # ffmpeg 이 없으면 OpenCV 내장(mp4v, 일부 휴대폰·브라우저 재생 불가)
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)); put = vw.write
        for k, jpg in frames:
            put(self._draw(cv2.imdecode(jpg, cv2.IMREAD_COLOR), job["boxes"], k, job["ev"], job["label"]))
        if self.ffmpeg: p.stdin.close(); p.wait()
        else: vw.release()
        self.db.update(job["id"], clip_path=str(job["sub"] / out.name))
        job["ev"]["boxes"] = None

    def status(self, extra):
        with self.lock:
            self._roll()
            st = dict(updated=now_str(), date=f"{self.day}", camera_id=self.cam, today=self.today, since_start=self.session, last_event=self.last, **extra)
        tmp = self.root / "status.json.tmp"
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(self.root / "status.json")

    def close(self):
        """호출 전에 판정 스레드가 끝나 있어야 한다 (남은 판정이 큐에 다 들어온 뒤). 큐·클립을 모두 쓰고 끝낸다."""
        self.stop_flag = True
        while self.is_alive():
            self.join(10)
            if self.is_alive(): self.log(f"기록 마무리 중... (대기 {self.q.qsize()}건, 클립 {len(self.pending)}건)")
