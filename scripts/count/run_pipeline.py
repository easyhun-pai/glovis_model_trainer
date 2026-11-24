"""COUNT 파이프라인 일괄 실행: 영상 → 추적 → 라인 통과 이벤트 → 비화물(번호판) 판정 → (GT 있으면) 평가.

사용:
  python scripts/count/run_pipeline.py data/COUNT/260701_s1_rtspcam.mp4 [더 많은 영상...] [--skip-extract]

단계별 스크립트:
  extract_tracks.py  → runs/count/<영상>/tracks.csv, crops/
  count_events.py    → events.csv        (영상별 기준선 data/COUNT/lines.json, 트랙 조각 연결, 중복 병합)
  diag_entries.py    → 미탐 후보 (우측 진입했는데 이벤트에 안 붙은 트랙)
  classify_events.py → event_cls.csv     (번호판 zero-shot → noncargo)
  eval_count.py      → data/COUNT/gt_v2/<영상>.csv 가 있을 때만
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
S = ROOT / "scripts" / "count"


def run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "_model_/count/yolo26m.pt"))
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    stems, preds = [], []
    for v in a.videos:
        v = Path(v); r = ROOT / "runs/count" / v.stem
        if not a.skip_extract or not (r / "tracks.csv").exists():
            run(S / "extract_tracks.py", v, "--model", a.model)
        run(S / "count_events.py", r)
        run(S / "diag_entries.py", r)
        run(S / "classify_events.py", r)
        if (ROOT / "data/COUNT/gt_v2" / f"{v.stem}.csv").exists():
            stems.append(v.stem); preds.append(str(r / "event_cls.csv"))
    if stems:
        run(S / "eval_count.py", *stems, "--pred", ",".join(preds))


if __name__ == "__main__":
    main()
