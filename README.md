# glovis_model_trainer

자동차 운반선 선적 현장의 고정 CCTV 영상으로 **두 개의 비전 모델**을 만들고 검증하는 학습 저장소.
두 과제는 같은 현장·같은 도구(Ultralytics YOLO, 사람이 확인하는 반자동 라벨링)를 쓰고, 폴더만 과제별로 나눴다.

| 과제 | 하는 일 | 모델 | 문서 |
|---|---|---|---|
| **VINCODE** | 차량 유리의 VIN 스티커 위치 탐지 → 바코드 디코더에 ROI 전달 | YOLO 1-class 탐지 (imgsz 1280, recall 우선) | [vincode/README.md](vincode/README.md) |
| **COUNT** | 램프 통과선을 넘는 화물차를 선적·하역별로 실시간 카운트, 세이프티카·작업차 제외 | 차량 탐지·추적(COCO yolo26m + ByteTrack) + 표식 탐지(번호판·경광등·창문번호, yolo26s) | [count/README.md](count/README.md) |

---

## 구조

```
glovis_model_trainer/
├── vincode/              VIN 스티커 탐지
│   ├── scripts/          프레임 선별 → 사전 라벨 → 분할 → 학습 → 비교 평가
│   ├── configs/          학습셋 구성 yaml
│   ├── models/           배포 모델 (.pt)
│   └── docs/             성능표·비교 영상
├── count/                선적·하역 카운트
│   ├── scripts/          트랙 체인·통과 판정·표식 라벨링·실시간 시뮬레이터
│   ├── web/              검수 웹페이지 (차량 이벤트·표식 판정·박스 수정)
│   ├── models/           표식 탐지 배포 모델 (.pt)
│   ├── deploy/           현장 실행 패키지 (RTSP 실시간 카운트 · DB · Ubuntu/Windows)
│   └── docs/
├── requirements.txt
│
│   ── 아래는 gitignore (로컬 전용) ──
├── data/VIN, data/COUNT  영상·프레임·크롭·학습셋·정답
├── runs/vincode, runs/count  학습·추론 산출물
└── _model_/              사전학습 가중치 캐시 (yolo26*, yoloe 등 자동 다운로드)
```

모든 스크립트는 **저장소 루트에서** 실행한다 (`python vincode/scripts/...`, `python count/scripts/...`).
경로는 스크립트 위치 기준으로 루트를 찾기 때문에 어디서 실행해도 같다.

## 공통 원칙

- **라벨링은 사람이 확인한다.** 모델·규칙이 초안을 만들고, 사람이 검수 도구(VisoLabel, 웹 검수 페이지)에서 고친 것만 정답으로 쓴다.
- **평가 영상은 학습에서 분리한다.** 날짜·세션 단위로 나누고, 최종 평가용 영상은 학습·튜닝·검토에 쓰지 않는다.
- **현장 비용 기준으로 판단한다.** VIN 은 미탐이 복구 불가라 recall 우선, COUNT 는 대수 정확도와 판정 지연(선 통과 후 1초 이내)을 함께 본다.

## 설치

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 버전에 맞춰 먼저
pip install -r requirements.txt
```

환경: Python 3.11, Ultralytics 8.4, PyTorch 2.6, RTX 2080 SUPER (8 GB).
