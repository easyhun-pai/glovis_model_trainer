"""표식 모델(r1/r2/r3)별 unseen 영상 정확도·처리시간 메타데이터 집계.

입력: runs/count/rt/<영상>_<모델>/{stats.json, eval_gt.json}  (realtime_sim + eval_rt 결과)
출력: runs/count/rt/model_compare/{model_compare.csv, model_compare.md}

  python count/scripts/model_compare.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RT = ROOT / "runs/count/rt"
OUT = RT / "model_compare"

MODELS = {
    "r1": "자동 초안(참조 1장) 학습",
    "r2": "개선 자동 초안(참조 4장·이동 규칙) 학습",
    "r3": "사용자 박스 수정 정답 + 자동 라벨 학습",
}
VIDEOS = [
    ("260701_s1_rtspcam", "0701 s1", "전체"),
    ("260701_s3_rtspcam", "0701 s3", "전체"),
    ("260701_s4_rtspcam", "0701 s4", "검수 구간(~03:47)"),
    ("260818_s3_rtspcam_검증용", "0818 s3 검증용", "전체(순서 비교)"),
]
# s1 정답 크롭 45장 기준 표식 P/R (eval_marks.py, IoU 0.5, conf 0.25)
MARK_PR = {
    "r1": {"plate": (0.95, 0.64), "beacon": (0.86, 0.76), "number": (0.45, 0.26)},
    "r2": {"plate": (0.95, 0.75), "beacon": (0.91, 0.87), "number": (0.45, 0.72)},
    "r3": {"plate": (0.95, 0.71), "beacon": (0.94, 0.85), "number": (0.47, 0.83)},
}


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row(model, key, name, scope):
    d = RT / f"{key}_{model}"
    st, ev = load(d / "stats.json"), load(d / "eval_gt.json")
    if st is None or ev is None:
        return None
    r = {"model": model, "video": name, "scope": scope}
    if "gt_len" in ev:  # 통과 순서 비교(검증용)
        gt, pc = ev["gt_counts"], ev["pred_counts"]
        r.update(gt_events=ev["gt_len"], correct=ev["n_match"],
                 missed=ev["n_del"], false=ev["n_ins"], wrong=ev["n_sub"],
                 load=f'{pc["load"]}/{gt["load"]}', unload=f'{pc["unload"]}/{gt["unload"]}',
                 exclude=f'{pc["exclude"]}/{gt["exclude"]}',
                 errors=" | ".join(ev["errors"]))
    else:
        wrong = len(ev["decision_wrong"])
        r.update(gt_events=ev["gt_events"], correct=ev["matched"] - wrong,
                 missed=len(ev["missed"]), false=len(ev["false"]), wrong=wrong,
                 load=f'{ev["load_cargo_pred"]}/{ev["load_cargo_gt"]}',
                 unload=f'{ev["unload_cargo_pred"]}/{ev["unload_cargo_gt"]}',
                 exclude=f'{ev["exclude_pred"]}/{ev["exclude_gt"]}',
                 errors=" | ".join(ev["decision_wrong"] + [f"미탐 {m}" for m in ev["missed"]]
                                   + [f"오탐 {f}" for f in ev["false"]]))
    r["acc"] = round(r["correct"] / r["gt_events"], 4)
    r.update(src_fps=st["fps_source"], proc_fps=st["processed_fps"],
             det_ms_p50=round(st["det_latency_s"]["p50"] * 1000, 1),
             mark_ms_p50=round(st["mark_latency_s"]["p50"] * 1000, 1),
             mark_ms_p95=round(st["mark_latency_s"]["p95"] * 1000, 1),
             gpu_mb=st["gpu_peak_alloc_mb"])
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for m in MODELS for k, n, s in VIDEOS if (r := row(m, k, n, s))]
    cols = ["model", "video", "scope", "gt_events", "correct", "acc", "missed", "false", "wrong",
            "load", "unload", "exclude", "src_fps", "proc_fps", "det_ms_p50", "mark_ms_p50",
            "mark_ms_p95", "gpu_mb", "errors"]
    with open(OUT / "model_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md = ["# 표식 모델별 unseen 영상 평가", "",
          "차량 탐지·추적(yolo26m + ByteTrack)과 판정 규칙은 모두 같고, 표식 모델만 바꿔 비교했다.",
          "모든 모델이 학습·튜닝에 쓰지 않은 영상이다. 정답은 사용자가 직접 센 것.", "",
          "## 모델 요약", "",
          "| 모델 | 학습 데이터 | 아키텍처 | 차량 정확(합계) | 판정 오류 | 미탐 | 오탐 |",
          "|---|---|---|---|---|---|---|"]
    for m, desc in MODELS.items():
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        g = sum(r["gt_events"] for r in rs)
        c = sum(r["correct"] for r in rs)
        md.append(f"| **{m}** | {desc} | yolo26s, 640, 40ep | {c}/{g} ({c / g:.1%}) | "
                  f"{sum(r['wrong'] for r in rs)} | {sum(r['missed'] for r in rs)} | "
                  f"{sum(r['false'] for r in rs)} |")
    md += ["", "## 영상별", "",
           "| 모델 | 영상 | 범위 | 정확 | 선적 화물 예측/정답 | 하역 화물 | 제외 | 오류 내용 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['video']} | {r['scope']} | {r['correct']}/{r['gt_events']} | "
                  f"{r['load']} | {r['unload']} | {r['exclude']} | {r['errors'] or '-'} |")
    md += ["", "## 표식 탐지 정확도 (0818 s1 정답 크롭 45장, IoU 0.5)", "",
           "| 모델 | 번호판 P / R | 경광등 P / R | 창문 번호 P / R |", "|---|---|---|---|"]
    for m, pr in MARK_PR.items():
        md.append(f"| {m} | " + " | ".join(f"{p:.2f} / {r:.2f}" for p, r in pr.values()) + " |")
    md += ["", "## 처리시간 (RTX 2080 SUPER)", "",
           "| 모델 | 영상 | 원본 fps | 처리 fps | 차량 탐지 p50 | 표식 판정 p50 / p95 | GPU 메모리 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['video']} | {r['src_fps']:.0f} | {r['proc_fps']:.1f} | "
                  f"{r['det_ms_p50']} ms | {r['mark_ms_p50']} / {r['mark_ms_p95']} ms | {r['gpu_mb']} MB |")
    rt = load(RT / "260818_s3_rtspcam_검증용_r3_realtime/stats.json")
    if rt:
        dl = rt["decision_latency_s"]
        md += ["", f"실시간 모드(r3, 0818 s3 검증용 15fps): 처리 {rt['processed_fps']} fps, "
                   f"프레임 버림 {rt['frames_dropped'] / rt['frames_received']:.1%}, "
                   f"선 넘음→판정 p50 {dl['p50']}s / p95 {dl['p95']}s / 최대 {dl['max']}s. "
                   "표식 모델 세 개는 구조·크기가 같아 처리시간 차이는 없다."]
    md += ["", "비교 정확: 차량 이벤트가 정답과 짝지어지고 화물/제외·방향 판정까지 맞은 대수. "
               "검증용 영상은 통과 순서 문자열 편집거리로 비교."]
    (OUT / "model_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
