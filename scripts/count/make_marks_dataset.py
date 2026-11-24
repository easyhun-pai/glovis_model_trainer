"""표식 탐지 YOLO 데이터셋: data/COUNT/marks/ds/{images,labels}/{train,val} + data.yaml
라벨: 세이프티·작업 크롭은 draft keep2 박스(또는 --labels 로 준 csv), 화물 크롭은 빈 라벨(배경).
이미지는 하드링크(같은 드라이브)로 복사 비용 없음.  사용: make_marks_dataset.py --train v.. --val v.. [--labels-dir 폴더]
"""
import argparse, os, shutil
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; MK = ROOT / "data/COUNT/marks"
ap = argparse.ArgumentParser(); ap.add_argument("--train", nargs="+"); ap.add_argument("--val", nargs="+")
ap.add_argument("--labels-dir", default=str(MK / "draft"), help="<영상>.csv (file,label,cls,x1..y2,W,H,keep2)"); ap.add_argument("--out", default=str(MK / "ds"))
a = ap.parse_args(); out = Path(a.out)
if out.exists(): shutil.rmtree(out)
for split, vids in (("train", a.train), ("val", a.val)):
    (out / "images" / split).mkdir(parents=True); (out / "labels" / split).mkdir(parents=True); n = nb = 0
    for v in vids:
        assert "검증용" not in v
        c = pd.read_csv(MK / "crops" / f"{v}.csv"); d = pd.read_csv(Path(a.labels_dir) / f"{v}.csv"); d = d[(d.keep2 == 1) & d.label.isin(["safety", "work"])]
        g = {f: x for f, x in d.groupby("file")}
        for r in c.itertuples():
            name = f"{v}__{r.file}"; os.link(MK / "crops" / v / r.file, out / "images" / split / name)
            lines = []
            if r.file in g:
                for q in g[r.file].itertuples():
                    lines.append(f"{int(q.cls)} {(q.x1 + q.x2) / 2 / q.W:.6f} {(q.y1 + q.y2) / 2 / q.H:.6f} {(q.x2 - q.x1) / q.W:.6f} {(q.y2 - q.y1) / q.H:.6f}")
            (out / "labels" / split / name.replace(".jpg", ".txt")).write_text("\n".join(lines)); n += 1; nb += len(lines)
    print(split, n, "images", nb, "boxes")
(out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n  1: beacon\n  2: number\n", encoding="utf8")
