"""검수 페이지에 항목 추가(기존 결과·클립은 그대로 두고 덧붙임).

사용: add_review_items.py <영상> <이벤트번호(events.csv 기준)>... [--group new] [--pred cargo]
- id 는 'N<frame>' (기존 E 번호는 events_v1 기준이라 섞이지 않게 함)
- 새 아틀라스 a/NNN.jpg 와 클립 묶음 clips/cNNN.mp4 를 만들고 items.json 에 c=-1(업로드 전) 로 기록
  → 클립 업로드 후 blobs 에 id 를 넣고 c 를 채운다.
"""
import argparse, glob, json, subprocess
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; WEB = ROOT / "runs/count/review/web"
FPS, HALF, W, H = 10, 1.5, 1280, 720; N = int(2 * HALF * FPS); TW, TH, COLS, ROWS = 480, 340, 8, 6
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("events", nargs="+")
ap.add_argument("--group", default="new"); ap.add_argument("--pred", nargs="*", default=[]); a = ap.parse_args()
M = json.load(open(WEB / "items.json", encoding="utf8")); v = a.video; run = ROOT / "runs/count" / v
ev = pd.read_csv(run / "events.csv", dtype={"event": str}).set_index("event"); pts = np.load(run / "pts.npy")
tr = pd.read_csv(run / "tracks_split.csv"); A, B = np.array(json.load(open(ROOT / "data/COUNT/lines.json"))[v], float)


def fit(im):
    t = np.full((TH, TW, 3), 16, np.uint8)
    if im is None: return t
    s = min(TW / im.shape[1], TH / im.shape[0]); q = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    y, x = (TH - q.shape[0]) // 2, (TW - q.shape[1]) // 2; t[y:y + q.shape[0], x:x + q.shape[1]] = q; return t


atlas = M["atlases"]; chunk = len(M["blobs"]); tiles = []; new = []; tmp = WEB / "clips/_t"; tmp.mkdir(parents=True, exist_ok=True); parts = []
cap = cv2.VideoCapture(str(ROOT / "data/COUNT" / f"{v}.mp4"))
for n, e in enumerate(a.events):
    r = ev.loc[e]; ch = [int(x) for x in str(r.chain).split("|")]; c = []
    for t in ch: c += sorted(glob.glob(str(run / "crops" / str(t % 1000000) / "*.jpg")))
    ims = [cv2.imread(p) for p in c]; i = int(np.argmax([m.shape[0] * min(m.shape[1], m.shape[0] * 1.7) for m in ims])) if ims else 0
    j = int(np.argmax([abs(k - i) for k in range(len(ims))])) if ims else 0
    tiles += [fit(ims[i] if ims else None), fit(ims[j] if ims and j != i else None)]
    t0 = pts[int(r.frame)]; idx = np.clip(np.searchsorted(pts, t0 + (np.arange(N) - N / 2) / FPS), 0, len(pts) - 1)
    path = tmp / f"add_{v}_{e}.mp4"; parts.append(path)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "33", "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0", "-bf", "0", "-pix_fmt", "yuv420p",
                          "-video_track_timescale", "90000", str(path)], stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (pts[idx[0]] - 5) * 1000)); fi = None
    # 정확한 프레임 위치를 위해 pts 로 맞춤: 키프레임부터 읽으며 시각 비교
    want = list(idx); k = 0; s = .5
    while k < N:
        ok, im = cap.read()
        if not ok: break
        tm = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        f = int(np.argmin(np.abs(pts - tm)))
        while k < N and want[k] <= f:
            o = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA); col = (0, 230, 255) if k >= N / 2 else (40, 40, 240)
            cv2.line(o, tuple((A * s).astype(int)), tuple((B * s).astype(int)), col, 3, cv2.LINE_AA)
            for b in tr[(tr.frame == f) & tr.tid.isin(ch)].itertuples():
                cv2.rectangle(o, (int(b.x1 * s), int(b.y1 * s)), (int(b.x2 * s), int(b.y2 * s)), (70, 220, 70), 2, cv2.LINE_AA)
                cv2.circle(o, (int((b.x1 + b.x2) / 2 * s), int(b.y2 * s)), 6, (0, 230, 255), -1, cv2.LINE_AA)
            p.stdin.write(o.tobytes()); k += 1
    p.stdin.close(); p.wait()
    sec = int(round(float(r.time_s)))
    new.append(dict(v=v, id=f"N{int(r.frame)}", g=a.group, t=f"{sec // 60:02d}:{sec % 60:02d}", d=r.dir, p=a.pred[n] if n < len(a.pred) else "", a=atlas, i=n, c=chunk, n=n))
while len(tiles) < COLS * ROWS: tiles.append(np.full((TH, TW, 3), 16, np.uint8))
cv2.imwrite(str(WEB / "a" / f"{atlas:03d}.jpg"), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 76])
lst = tmp / "list.txt"; lst.write_text("".join(f"file '{q.as_posix()}'\n" for q in parts), encoding="utf8")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(WEB / "clips" / f"c{chunk:03d}.mp4")], check=True)
M["items"] += new; M["atlases"] = atlas + 1
json.dump(M, open(WEB / "items.json", "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
print(json.dumps(new, ensure_ascii=False)); print("chunk", WEB / "clips" / f"c{chunk:03d}.mp4", "(blobs 에 업로드 id 추가 필요)")
