"""COUNT 4단계: 이벤트(체인)별 '비화물' 판정 — zero-shot 번호판 검출 (YOLOE 텍스트 프롬프트).

근거 (260701 GT 289건 검증):
  - 화물차(선적/하역 대상)는 번호판이 없다. 세이프티카/작업차는 번호판이 있다.
  - 차량 전체 외형 임베딩(DINOv2) 분류는 '흰색=세이프티'로 학습돼 흰/은색 화물차를 오판 (17/169).
  - 번호판 zero-shot 은 학습 없이 오판 3/169 (나머지는 이 스크립트의 규칙으로 처리).

절차:
  1. 이벤트 체인의 검출 박스를 --every 간격으로 샘플링, 순차 디코딩으로 크롭(상단 여유 포함)
  2. 크롭에 YOLOE('license plate') → 번호판 박스를 프레임 좌표로 환산
  3. 오귀속 방지: 번호판 중심을 포함하는 같은 프레임의 차량 박스가 여럿이면
     박스 하단(y2)이 번호판 중심보다 아래이면서 가장 가까운 차량에 귀속. 다른 차량이면 버린다.
  4. 체인 max conf >= --thr → noncargo
  5. 체인 차량 클래스 다수결이 truck/bus 면 work_cand=1 (지게차·트럭 등, 별도 보고)

출력: runs/count/<영상>/event_cls.csv  (video, event, dir, noncargo, plate_conf, plate_frames, n, work_cand, major_cls)
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[2]


def owner(px, py, boxes):
    """번호판 중심 (px,py) 의 소유 차량 tid. boxes: [(tid,x1,y1,x2,y2)]"""
    c = [(b[4] - py, b[0]) for b in boxes if b[1] <= px <= b[3] and b[2] <= py <= b[4]]
    c = [x for x in c if x[0] >= 0]
    return min(c)[1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default=str(ROOT / "_model_/count/yoloe-11l-seg.pt"))
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=150)
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--det-conf", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--edge", type=float, default=3, help="크롭 경계 몇 px 이내를 '닿음'으로 봄 (최소값)")
    ap.add_argument("--edge-rel", type=float, default=0.03, help="크롭 크기 대비 경계 여백 비율 (s2 E50: 5px/418px)")
    ap.add_argument("--min-aspect", type=float, default=1.5, help="번호판 박스 최소 가로/세로 비")
    a = ap.parse_args()
    run = Path(a.run); meta = json.load(open(run / "meta.json"))
    ev = pd.read_csv(run / "events.csv"); tr = pd.read_csv(run / "tracks_split.csv")
    tr_all = tr; tr = tr[(tr.y2 - tr.y1) >= a.min_h]
    tid2ev = {}
    for r in ev.itertuples():
        for t in str(r.chain).split("|"): tid2ev.setdefault(int(t), r.event)
    sel = tr[tr.tid.isin(tid2ev)].copy(); sel["event"] = sel.tid.map(tid2ev)
    sel = sel[sel.groupby("tid").cumcount() % a.every == 0]
    # 귀속 판정엔 높이 필터 없이 모든 차량 박스 사용 (잘린 옆 차 박스가 작아도 포함돼야 함)
    by_frame = {f: g for f, g in tr_all.groupby("frame")}
    m = YOLOE(a.model); names = ["license plate"]; m.set_classes(names, m.get_text_pe(names))
    evid = run / "plate_evidence"; evid.mkdir(exist_ok=True)
    best_ev = {}  # event -> (conf, crop_with_box)

    FW, FH = meta["W"], meta["H"]
    cap = cv2.VideoCapture(meta["video"]); fi = -1; want = set(sel.frame); last = max(want) if want else -1
    rec = []  # (event, conf)
    pend = []  # (crop, event, tid, frame, ox, oy)

    def flush():
        if not pend: return
        res = m.predict([p[0] for p in pend], conf=a.det_conf, verbose=False)
        for (crop, e, tid, f, ox, oy), r in zip(pend, res):
            best, bb = 0.0, None; ch, cw = crop.shape[:2]
            # 크롭 변이 프레임 경계와 같으면(차가 화면 밖으로 잘림) 그 변은 경계 규칙 예외
            fl, ft, fr_, fb = ox <= 1, oy <= 1, ox + cw >= FW - 1, oy + ch >= FH - 1
            boxes = [(int(b.tid), b.x1, b.y1, b.x2, b.y2) for b in by_frame[f].itertuples()]
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                # (1) 크롭 모서리(프레임 경계가 아닌 변 2개 이상)에 걸린 번호판 = 옆 차에서 잘려 들어온 것
                #     (s3 823s: 옆 세이프티카 번호판). 한 변만 닿는 건 허용 — 램프에서 나오는 L2R 차는
                #     박스가 차 일부만 감싸 자기 번호판이 박스 끝에 붙는다.
                ex, ey = max(a.edge, a.edge_rel * cw), max(a.edge, a.edge_rel * ch)
                touch = (x1 <= ex and not fl) + (y1 <= ey and not ft) + \
                        (x2 >= cw - ex and not fr_) + (y2 >= ch - ey and not fb)
                if touch >= 2: continue
                # (2) 번호판은 가로로 길다. 정사각형에 가까운 박스 = 후미등/반사판 오검 (s4 742s)
                if (x2 - x1) < a.min_aspect * (y2 - y1): continue
                px, py = ox + (x1 + x2) / 2, oy + (y1 + y2) / 2
                if owner(px, py, boxes) == tid and cf > best: best, bb = cf, (x1, y1, x2, y2)
            rec.append((e, best))
            if bb and best > best_ev.get(e, (0,))[0]:
                im = crop.copy(); x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(im, f"{best:.2f}", (x1, max(20, y1 - 6)), 0, 1.0, (0, 0, 255), 2)
                best_ev[e] = (best, im)
        pend.clear()

    groups = {f: g for f, g in sel.groupby("frame")}
    while fi < last:
        if not cap.grab(): break
        fi += 1
        if fi not in want: continue
        im = cap.retrieve()[1]; H, W = im.shape[:2]
        for b in groups[fi].itertuples():
            h = b.y2 - b.y1; y1 = max(0, int(b.y1 - a.pad * h)); x1 = max(0, int(b.x1))
            pend.append((im[y1:int(b.y2), x1:min(W, int(b.x2))].copy(), b.event, int(b.tid), fi, x1, y1))
        if len(pend) >= 32: flush()
    flush()
    for f in evid.glob("*.jpg"): f.unlink()
    for e, (c, im) in best_ev.items():
        if c >= a.thr: cv2.imwrite(str(evid / f"E{e}_{c:.2f}.jpg"), im)

    r = pd.DataFrame(rec, columns=["event", "conf"])
    agg = r.groupby("event").agg(plate_conf=("conf", "max"), plate_frames=("conf", lambda s: int((s >= a.thr).sum())),
                                 n=("conf", "size")).reset_index()
    cls = sel.groupby("event").cls.agg(lambda s: int(s.mode().iloc[0])).rename("major_cls").reset_index()
    out = ev[["event", "dir", "time_s"]].merge(agg, on="event", how="left").merge(cls, on="event", how="left")
    out = out.fillna({"plate_conf": 0.0, "plate_frames": 0, "n": 0, "major_cls": 2})
    out["noncargo"] = (out.plate_conf >= a.thr).astype(int)
    out["work_cand"] = out.major_cls.isin([5, 7]).astype(int)
    out.insert(0, "video", run.name)
    out.to_csv(run / "event_cls.csv", index=False)
    for d, g in out.groupby("dir"):
        print(f"{run.name} [{d}] events={len(g)} 비화물(번호판)={int(g.noncargo.sum())} "
              f"화물={int((g.noncargo == 0).sum())} (그중 truck/bus 클래스 {int(((g.noncargo == 0) & (g.work_cand == 1)).sum())})")


if __name__ == "__main__":
    main()
