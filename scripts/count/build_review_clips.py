"""검수 페이지용 통과 클립 생성: items.json 의 차량마다 통과 시점 ±1.5초 영상(1280x720, 10fps).

표시: 빨간 선 = 통과 기준선(차량이 넘는 순간부터 노란색), 초록 박스 = 이 항목의 차량, 점 = 판정 지점.
출력: runs/count/review/web/clips/c###.mp4 (최대 24대씩 이어 붙임), clips.json {영상/id: [묶음, 순번]}
홀드아웃은 items.json 에 없으므로 처리되지 않는다.   사용: build_review_clips.py [--limit N]
"""
import argparse, json, subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; WEB = ROOT / "runs/count/review/web"; OUT = WEB / "clips"; TMP = OUT / "_t"
FPS, HALF, W, H = 10, 1.5, 1280, 720; N = int(2 * HALF * FPS); MAX_N, MAX_B = 24, 17 * 1048576
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
TMP.mkdir(parents=True, exist_ok=True)
items = json.load(open(WEB / "items.json", encoding="utf8"))["items"]
if a.limit: items = items[:a.limit]
lines = json.load(open(ROOT / "data/COUNT/lines.json", encoding="utf8"))


def enc(path):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "33", "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0", "-bf", "0",
                             "-threads", "2", "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", str(path)], stdin=subprocess.PIPE)


by = defaultdict(list)
for k, it in enumerate(items): by[it["v"]].append((k, it))
for v, its in by.items():
    run = ROOT / "runs/count" / v; pts = np.load(run / "pts.npy"); tr = pd.read_csv(run / "tracks_split.csv"); ev = pd.read_csv(run / "events.csv", dtype={"event": str}).set_index("event")
    A, B = np.array(lines[v], float); AB = B - A; frames = {f: g for f, g in tr.groupby("frame")}; need = defaultdict(list); clips = {}
    for k, it in its:
        if it["g"] == "miss":
            tid = int(it["id"][1:]); g = tr[tr.tid == tid]; bx = (g.x1 + g.x2) / 2
            f0 = int(g.frame.iloc[int(np.argmin(np.abs(AB[0] * (g.y2 - A[1]) - AB[1] * (bx - A[0]))))]); ch = {tid}
        else:
            f0 = int(ev.frame[it["id"]]); ch = {int(x) for x in str(ev.chain[it["id"]]).split("|")}
        t0 = pts[min(f0, len(pts) - 1)]; idx = np.clip(np.searchsorted(pts, t0 + (np.arange(N) - N / 2) / FPS), 0, len(pts) - 1)
        clips[k] = dict(ch=ch, n=0, p=None, path=TMP / f"{k:04d}.mp4", last=None)
        for j, f in enumerate(idx): need[int(f)].append((k, j))
    cap = cv2.VideoCapture(str(ROOT / "data/COUNT" / f"{v}.mp4")); fi = -1; end = max(need)
    while fi < end:
        if not cap.grab(): break
        fi += 1
        if fi not in need: continue
        im = cv2.resize(cap.retrieve()[1], (W, H), interpolation=cv2.INTER_AREA)
        for k, j in sorted(need[fi]):
            c = clips[k]; o = im.copy(); s = .5; col = (0, 230, 255) if j >= N / 2 else (40, 40, 240)
            cv2.line(o, tuple((A * s).astype(int)), tuple((B * s).astype(int)), col, 3, cv2.LINE_AA)
            if fi in frames:
                for b in frames[fi][frames[fi].tid.isin(c["ch"])].itertuples():
                    cv2.rectangle(o, (int(b.x1 * s), int(b.y1 * s)), (int(b.x2 * s), int(b.y2 * s)), (70, 220, 70), 2, cv2.LINE_AA)
                    cv2.circle(o, (int((b.x1 + b.x2) / 2 * s), int(b.y2 * s)), 6, (0, 230, 255), -1, cv2.LINE_AA)
            if c["p"] is None: c["p"] = enc(c["path"])
            c["p"].stdin.write(o.tobytes()); c["n"] += 1; c["last"] = o
            if c["n"] == N: c["p"].stdin.close(); c["p"].wait(); c["p"] = None
    for c in clips.values():  # 영상 끝에서 프레임이 모자라면 마지막 장면으로 채움
        if c["p"] is not None:
            for _ in range(N - c["n"]): c["p"].stdin.write(c["last"].tobytes())
            c["p"].stdin.close(); c["p"].wait()
    print(v, len(its), flush=True)

for f in OUT.glob("c*.mp4"): f.unlink()
paths = [TMP / f"{k:04d}.mp4" for k in range(len(items))]; groups, cur, size = [], [], 0
for k, p in enumerate(paths):
    b = p.stat().st_size
    if cur and (len(cur) == MAX_N or size + b > MAX_B): groups.append(cur); cur, size = [], 0
    cur.append(k); size += b
if cur: groups.append(cur)
index = {}
for c, g in enumerate(groups):
    lst = TMP / "list.txt"; lst.write_text("".join(f"file '{paths[k].as_posix()}'\n" for k in g), encoding="utf8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(OUT / f"c{c:03d}.mp4")], check=True)
    for n, k in enumerate(g): index[f'{items[k]["v"]}/{items[k]["id"]}'] = [c, n]
json.dump(dict(dur=2 * HALF, index=index), open(WEB / "clips.json", "w", encoding="utf8"), separators=(",", ":"))
S = [f.stat().st_size for f in OUT.glob("c*.mp4")]
print("clips", len(items), "chunks", len(S), "total MB", round(sum(S) / 1048576, 1), "max MB", round(max(S) / 1048576, 2))
