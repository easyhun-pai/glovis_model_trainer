"""로컬 DB(logs/count.db) 조회·내보내기.

  python app/db_tool.py summary                        일별 대수
  python app/db_tool.py export --from 2026-11-01 --to 2026-11-30 --out nov.csv   기간 내 차량 목록 (csv 또는 .jsonl)
  python app/db_tool.py export --unsynced --out batch.jsonl --mark-synced         중앙 DB 로 아직 안 올린 행만 내보내고 표시

중앙 DB 연동 시: 같은 스키마(storage.SCHEMA)를 만들고, 주기적으로 --unsynced 내보내기 → 적재 → --mark-synced.
미디어 파일은 *_path 컬럼(logs 기준 상대 경로)으로 찾아 파일 서버/오브젝트 스토리지에 함께 올린다.
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 패키지 루트


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "export"])
    ap.add_argument("--db", default=str(ROOT / "logs" / "count.db"))
    ap.add_argument("--from", dest="d0"); ap.add_argument("--to", dest="d1")
    ap.add_argument("--unsynced", action="store_true"); ap.add_argument("--mark-synced", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not Path(a.db).exists(): sys.exit(f"DB 없음: {a.db}")
    c = sqlite3.connect(a.db); c.row_factory = sqlite3.Row

    if a.cmd == "summary":
        rows = c.execute("SELECT * FROM daily_counts ORDER BY day, camera_id").fetchall()
        print(f"{'날짜':<12}{'카메라':<14}{'선적 화물':>8}{'하역 화물':>8}{'제외':>6}{'합계':>6}")
        for r in rows: print(f"{r['day']:<12}{r['camera_id']:<14}{r['load_cargo']:>8}{r['unload_cargo']:>8}{r['excluded']:>6}{r['total']:>6}")
        return

    q, args = "SELECT * FROM events WHERE 1=1", []
    if a.d0: q += " AND substr(crossed_at,1,10) >= ?"; args.append(a.d0)
    if a.d1: q += " AND substr(crossed_at,1,10) <= ?"; args.append(a.d1)
    if a.unsynced: q += " AND synced = 0"
    rows = [dict(r) for r in c.execute(q + " ORDER BY id", args).fetchall()]
    out = Path(a.out or "events_export.csv")
    if out.suffix == ".jsonl":
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    else:
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            if rows: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.mark_synced and rows:
        c.executemany("UPDATE events SET synced=1 WHERE id=?", [(r["id"],) for r in rows]); c.commit()
    print(f"{len(rows)}건 → {out}" + (" (synced 표시)" if a.mark_synced else ""))


if __name__ == "__main__":
    main()
