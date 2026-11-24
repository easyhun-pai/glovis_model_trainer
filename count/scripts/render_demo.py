"""데모 영상 렌더 (realtime_sim.py --render 를 차례로 호출, GPU 작업은 한 번에 하나).

montage: 구간 여러 개를 렌더해 ffmpeg 로 이어 붙인다.
  python count/scripts/render_demo.py montage --out runs/count/rt/demo/montage_r3.mp4 --seg 260818_s1_rtspcam:1100:20 --seg 260701_s3_rtspcam:816:20 ...
errors: 각 실행 폴더의 eval_errors.csv 마다 오류 시각 -4초~+4초 클립. 추적 상태를 전체 실행과 맞추려고 30초 앞부터 처리하고 기록은 -4초부터.
  python count/scripts/render_demo.py errors --runs runs/count/rt/260701_s1_rtspcam_r3 ... --out-dir runs/count/rt/demo/errors
"""
import argparse, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "count/scripts/realtime_sim.py"
MARK = "runs/count/marks/r3/weights/best.pt"
TYPE_KO = {"missed": "미탐", "false": "오탐", "wrong": "판정 오류", "substitution": "순서 불일치(대체)", "insertion": "추가 예측", "deletion": "미탐(정답에만)"}


def video_path(stem):
    return ROOT / "data/COUNT" / f"{stem}.mp4"


def line_key(stem):
    return "260818_s2_rtspcam" if stem.startswith("260818_s3") else stem


def sim(stem, start, secs, render, tmp, extra):
    cmd = [sys.executable, str(SIM), str(video_path(stem)), "--no-drop", "--mark-model", MARK, "--cross-grace", "0.5", "--line-key", line_key(stem),
           "--start", f"{start:.2f}", "--max-seconds", f"{secs:.2f}", "--render", str(render), "--out", str(tmp), *extra]
    if "검증용" in stem: cmd.append("--final-holdout")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode: print(r.stdout[-1500:], r.stderr[-3000:]); raise SystemExit(f"render 실패: {stem} {start}")


def mmss(t): return f"{int(t // 60):02d}:{int(t % 60):02d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["montage", "errors"])
    ap.add_argument("--seg", action="append", default=[]); ap.add_argument("--out"); ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out-dir", default="runs/count/rt/demo/errors"); ap.add_argument("--pre", type=float, default=30); ap.add_argument("--win", type=float, default=4)
    a = ap.parse_args()
    if a.mode == "montage":
        out = ROOT / a.out; tmpd = out.parent / "_seg"; tmpd.mkdir(parents=True, exist_ok=True); parts = []
        for i, sg in enumerate(a.seg):
            stem, t0, L = sg.split(":"); t0, L = float(t0), float(L)
            title = f"{stem.replace('_rtspcam', '').replace('_검증용', ' (검증용)')}  {mmss(t0)}~{mmss(t0 + L)}"
            p = tmpd / f"seg{i}.mp4"
            # 15초 앞부터 처리해 추적을 안정시키고, 기록은 t0 부터
            sim(stem, max(0, t0 - 15), L + min(15, t0), p, tmpd / f"run{i}", ["--render-from", f"{t0:.2f}", "--render-title", title])
            parts.append(p); print("seg", i, p, f"{p.stat().st_size / 2**20:.1f} MB", flush=True)
        # 영상마다 fps 가 달라(15/20) concat demuxer 로 붙이면 길이가 틀어짐 → concat 필터로 15fps 정규화 후 재인코딩
        fc = "".join(f"[{i}:v]fps=15,setpts=PTS-STARTPTS[v{i}];" for i in range(len(parts))) + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1[v]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *sum([["-i", str(p)] for p in parts], []), "-filter_complex", fc, "-map", "[v]",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        print("montage", out, f"{out.stat().st_size / 2**20:.1f} MB")
        return
    od = ROOT / a.out_dir; od.mkdir(parents=True, exist_ok=True); idx = []
    for run in a.runs:
        run = ROOT / run; stem = run.name.replace("_r3", "")
        E = pd.read_csv(run / "eval_errors.csv", encoding="utf-8-sig", keep_default_na=False)
        for e in E.itertuples():
            t = float(e.time_s); start = max(0.0, t - a.win - a.pre)
            banner = f"{stem.replace('_rtspcam', '')} {mmss(t)} | {TYPE_KO.get(e.type, e.type)} | 예측 {e.pred} vs 정답 {e.gt}" + (f" | {e.reason}" if e.reason else "")
            f = od / f"{stem}_{mmss(t).replace(':', '')}_{e.type}.mp4"
            sim(stem, start, (t + a.win) - start, f, od / "_tmp" / stem,  # pts.npy 캐시가 영상별이어야 함
                ["--render-from", f"{max(0, t - a.win):.2f}", "--render-banner", banner,
                                                                  "--render-crf", "28"])
            idx.append(dict(video=stem, time=mmss(t), time_s=t, type=e.type, dir=e.dir, pred=e.pred, gt=e.gt, reason=e.reason, file=str(f.relative_to(ROOT)),
                            size_mb=round(f.stat().st_size / 2**20, 2)))
            print(idx[-1], flush=True)
    pd.DataFrame(idx).to_csv(od / "index.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
