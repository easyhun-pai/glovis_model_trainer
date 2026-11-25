# COUNT 현장 실행 패키지 v1.2-rc2

램프 CCTV(RTSP)를 받아 **통과 기준선을 넘는 화물차를 선적·하역별로 실시간으로 센다.**
세이프티카·작업차는 표식(번호판·경광등·창문 번호)으로 가려내 "제외"로 따로 센다.

```
카메라(RTSP) → 차량 탐지·추적 (yolo26m + ByteTrack)
            → 기준선 통과 순간 1대 (야드→배 = 선적, 배→야드 = 하역)
            → 접근 중 차량 크롭 최대 8장에서 표식 탐지 (marks_r3)
            → 차와 같이 움직이는 표식이 2장 이상 → 제외 / 아니면 화물
```

선을 넘은 뒤 판정까지 보통 0.3초, 최대 1초 안이다.

## 폴더 구성

```
COUNT_deploy/
├── README.md             이 설명서
├── install.sh            ① 처음 한 번: 설치 (Ubuntu 22.04 / 24.04)
├── run.sh                ② 실행 (시작할 때 기준선 확인 화면)
├── setup_line.sh            기준선을 새로 긋고 실행 (화각이 바뀌었을 때)
├── test_video.sh            녹화 영상으로 시험 실행: ./test_video.sh 영상.mp4 (설정·기록에 영향 없음)
├── db_summary.sh            일별 대수 보기
├── config.json           카메라 주소 · 기준선 · 동작 설정
├── linux/
│   ├── install_autostart.sh  로그인하면 자동 실행 등록 (해제: --remove)
│   └── run_forever.sh        오류로 꺼지면 10초 뒤 다시 켬
├── windows/              Windows PC 용 (install.bat · run.bat 등, 같은 순서)
├── models/
│   ├── yolo26m.pt        차량 탐지 (공개 COCO 모델)
│   └── marks_r3.pt       표식 탐지 (번호판·경광등·창문 번호, 현장 데이터로 학습)
├── app/                  프로그램 (손대지 않음)
│   ├── count_live.py     실행 진입점
│   ├── core.py           판정 엔진 (탐지·추적·통과·표식 판정)
│   ├── line_setup.py     기준선 설정 화면
│   ├── storage.py        DB · 스크린샷 · 클립 저장
│   ├── db_tool.py        DB 조회 · 내보내기
│   └── requirements.txt
└── logs/                 실행하면 기록이 쌓이는 곳 (DB · 사진 · 클립 · CSV · 로그)
```

설치하면 `.venv/` (Python 3.11 가상환경) 가 생긴다.

## 필요 환경

- **Ubuntu 22.04 또는 24.04 데스크톱** (모니터 연결), NVIDIA GPU + 드라이버 (535 이상 권장, `nvidia-smi` 가 되면 됨)
  - RTX 2080 SUPER 에서 검증: 20fps 카메라 실시간 처리, GPU 메모리 약 0.4 GB
- 설치할 때 인터넷 필요 (PyTorch · TensorRT 등 약 5 GB 다운로드), sudo 권한 (한글 폰트 등 시스템 패키지)
- Python 은 따로 준비하지 않아도 된다. `install.sh` 가 uv 로 3.11 을 받는다
  (22.04 기본 3.10 은 일부 패키지 미지원, 24.04 는 시스템 Python 에 pip 설치가 막혀 있음).
- GPU 가 없으면 CPU 로 돌지만 실시간 처리는 어렵다.
- Windows 에서는 `windows/` 의 bat 파일을 같은 순서로 쓴다 (Python 3.11 직접 설치 필요).

## 처음 설정

```bash
unzip COUNT_deploy_*.zip && cd COUNT_deploy
chmod +x *.sh linux/*.sh     # zip 을 풀면 실행 권한이 빠질 수 있음
./install.sh                 # 10~20분 (다운로드 속도에 따라)
nano config.json             # "source" 에 카메라 주소
./run.sh
```

`config.json` 의 카메라 주소 예: `"source": "rtsp://아이디:비밀번호@192.168.0.10:554/stream1"`
(영상 파일 경로나 USB 카메라 번호 `"0"` 도 된다)

## 처음 실행 시 TensorRT 엔진 생성

`./run.sh` 를 처음 실행하면 이 PC 의 GPU 에 맞춘 TensorRT 엔진 2개를 만든다 (차량 탐지 · 표식 탐지, **합계 약 10분**, 한 번만).
엔진은 `models/` 에 `..._<GPU이름>_trt<버전>.engine` 으로 저장되고 다음 실행부터 바로 쓴다. GPU 나 TensorRT 버전이 바뀌면 자동으로 새로 만든다.
엔진을 만들거나 읽다 실패하면 그 엔진을 `.bad` 로 치우고 PyTorch(.pt) 로 실행한다 (조금 느림). 끄려면 `config.json` 의 `"use_tensorrt": false`.

## 자동 실행 (PC 를 켜면 바로)

```bash
./linux/install_autostart.sh          # 로그인하면 터미널 창과 함께 자동 실행 (해제: --remove)
```

- PC 가 켜지면 바로 돌게 하려면 Ubuntu [설정 > 사용자] 에서 **자동 로그인**, [설정 > 전원] 에서 **화면 끄기·절전 끄기**.
- 오류로 꺼지면 10초 뒤 다시 켠다 (`logs/restart.log`). 사람이 Q 로 끄면 다시 켜지 않는다.
- 시작 때 기준선 확인 화면은 15초 뒤 저장된 선으로 자동 진행한다.

## 기준선 긋기

카메라 첫 화면 위에서:

1. **선의 시작점 클릭 → 끝점 클릭.** 차량이 넘어가는 배수로 등을 따라 램프 폭 전체에 걸쳐 긋는다.
   차의 **바퀴 쪽(박스 아래 가운데)** 이 이 선을 넘을 때 센다.
2. **야드 쪽을 한 번 클릭.** 선적 차량이 출발하는 쪽(배 반대편). 화면에 "야드 쪽 / 배 쪽" 이 표시된다.
3. **Enter** 로 시작. 다시 그리려면 **R**, 그만두려면 **Esc**.

- 끝점을 화면 왼쪽·오른쪽 가장자리에 바짝(2% 안) 찍으면 선이 화면 밖까지 연장된다.
  카메라에 가까운 차가 선을 넘기 전에 화면 밖으로 잘려 나가는 화각이면 이렇게 긋는다.
- 그은 선은 `config.json` 에 저장된다. 다음 실행부터는 저장된 선을 보여 주고 **15초 뒤 자동 시작**한다
  (PC 재부팅 후 무인 재시작 대비). 화각이 바뀌었으면 그때 **R** 을 누르거나 `./setup_line.sh` 를 쓴다.

## 실행 중

- 화면: 기준선(하늘색), 차량 박스(회색 판정 전 · 초록 화물 · 빨강 제외), 오른쪽 아래에 **오늘 누적 대수**.
- **Q** 또는 Esc 로 종료. 진행 중인 판정을 마무리하고 끝낸다.
- 카메라가 끊기면 5초마다 다시 연결한다.
- 화면 없이 돌리려면 `./run.sh --no-show` (기준선이 저장돼 있어야 함).

## 기록 (`logs/`)

차량 1대가 판정되면 **DB 1행 + 사진·클립**이 남는다.

| 파일 | 내용 |
|---|---|
| `count.db` | 로컬 DB (SQLite). 아래 스키마 |
| `media/YYYYMMDD/<번호>_<시각>_<방향>_<판정>_shot.jpg` | 통과 순간 전체 화면 (기준선·차량 박스·번호·판정 표시) |
| `media/.../_clip.mp4` | 통과 전후 1.5초 클립 (총 3초, H.264 — 휴대폰·브라우저 재생) |
| `media/.../_crop.jpg` | 표식 판정에 쓴 차량 크롭 (원본 해상도) |
| `events_YYYYMMDD.csv` | DB 와 같은 내용의 사람용 사본 (엑셀) |
| `status.json` | 오늘 누적 대수, 처리 fps, 버린 프레임 비율, 마지막 차량. 5초마다 갱신 |
| `run_YYYYMMDD.log` | 실행 로그 |

프로그램을 다시 켜도 오늘 누적은 DB 에서 이어서 센다. 사진·클립은 차량당 약 0.5 MB (하루 500대면 약 250 MB).
조회: `./db_summary.sh` (일별 대수), 기간 내보내기는 `.venv/bin/python app/db_tool.py export --from 2026-11-01 --out 11월.csv`.

### DB 스키마

`events` — 차량 1대 1행

| 컬럼 | 내용 |
|---|---|
| `id` | 차량 번호 (전체 기간 고유, 파일명 앞 6자리와 같음) |
| `camera_id`, `run_id` | 카메라 (config 의 `camera_id`), 실행 회차 |
| `crossed_at` / `crossed_ts` | 기준선 통과 시각 (현지 `YYYY-MM-DD HH:MM:SS.mmm` / UNIX 초) |
| `direction` | `load` 선적(야드→배) / `unload` 하역(배→야드) |
| `vehicle_class` | `cargo` 화물 / `exclude` 제외(세이프티·작업차) |
| `marks_plate` · `marks_beacon` · `marks_number` | 표식이 보인 크롭 수 (판정 근거) |
| `n_crops`, `reason` | 판정에 쓴 크롭 수, 근거 문구 |
| `decided_at`, `latency_s` | 판정 확정 시각, 통과→판정 초 |
| `screenshot_path` · `clip_path` · `crop_path` | 미디어 파일 (`logs/` 기준 상대 경로) |
| `synced` | 중앙 DB 업로드 여부 (0/1) |

`runs` — 실행 1회 1행 (시작·종료 시각, 입력, 해상도·fps, 기준선, 모델 이름). `daily_counts` — 일별 대수 뷰.

### 중앙 DB 연동 (나중에)

표준 SQL 타입만 써서 PostgreSQL·MySQL·MSSQL 에 같은 테이블을 그대로 만들 수 있다 (`app/storage.py` 의 `SCHEMA`).
가장 간단한 방법은 주기적으로 안 올린 행만 내보내 적재하는 것:

```
.venv/bin/python app/db_tool.py export --unsynced --out batch.jsonl --mark-synced
```

사진·클립은 `*_path` 로 찾아 파일 서버나 오브젝트 스토리지에 같이 올리고, 중앙 DB 에는 그 주소를 넣는다.

## 설정 (`config.json`)

| 키 | 기본 | 설명 |
|---|---|---|
| `source` | | RTSP 주소 / 영상 파일 / 웹캠 번호 |
| `fps` | null | 카메라 fps. null 이면 스트림 정보 사용 (이상하면 직접 입력) |
| `show_window` | true | 화면 표시 |
| `confirm_line_on_start` | true | 시작할 때 기준선 확인 화면 |
| `line_confirm_timeout_s` | 15 | 확인 화면 자동 시작까지 초 |
| `line` | null | 기준선 (설정 화면에서 자동 저장) |
| `camera_id` | ramp_cam1 | 카메라 이름 (DB·여러 대 운용 시 구분) |
| `save_screenshots` · `save_clips` · `save_crops` | true | 통과 스크린샷 · 전후 클립 · 판정 크롭 저장 |
| `clip_pre_s` · `clip_post_s` | 1.5 · 1.5 | 클립 길이 (통과 전·후 초) |
| `media_width` | 1280 | 스크린샷·클립 가로 해상도 |
| `use_tensorrt` | true | TensorRT 엔진 사용 (없으면 처음 실행 때 생성) |
| `kalman_gap` | true | 처리가 밀려 프레임을 건너뛰었을 때 추적 예측을 실제 경과 시간만큼 보정 |
| `replay_guard` | true | RTSP 재접속 때 몇 초 전 영상이 다시 들어오면 중복으로 세지 않음 |
| `params` | | 판정 파라미터. 검증된 값이므로 바꾸지 않는다 |

`params` 의 픽셀 값은 2560x1440 기준이며, 카메라 해상도가 다르면 가로 폭 비율로 자동 환산된다.

## 검증 결과 (학습에 쓰지 않은 영상)

| 영상 | 결과 |
|---|---|
| 2026-07-01 s1·s3·s4 (20fps) | 정답 206대 중 204대 정확 |
| 2026-08-18 s3 (15fps, 최종 검증용) | 103대 중 101대 정확 (선적 34/33, 하역 34/36, 제외 35/34) |

알려진 약점

- 반대 방향으로 엇갈리는 차량끼리 추적이 섞이는 경우
- 번호판이 뒤에만 있는 작업 트럭 (선을 넘기 전에 표식이 안 보임)
- 차가 지나갈 때 램프 구조물 벽면이 잠깐 차량으로 잡혀 1대가 더 세지는 경우가 드물게 있음 (5분에 0~1회). 클립으로 확인 가능

## 버전

| 버전 | 날짜 | 내용 |
|---|---|---|
| 1.2-rc2 | 2026-11-25 | Ubuntu 지원(install.sh · 자동 실행), TensorRT 가속, 추적 보정(건너뛴 프레임만큼 칼만 예측, 램프 벽면 오검출로 id 가 넘어가는 경우 분리), 재접속·종료·자정 처리 수정. 검증 진행 중 |
| 1.1 | 2026-11-25 | 로컬 DB(SQLite), 통과 순간 스크린샷 · 전후 1.5초 클립 저장, 폴더 정리 |
| 1.0 | 2026-11-25 | 첫 배포: RTSP 실시간 카운트, 시작 시 기준선 설정 화면 |
