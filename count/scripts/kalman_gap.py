"""ByteTrack 칼만 예측의 프레임 간격 보정.

문제: Ultralytics ByteTrack 은 update() 1회 = 1프레임으로 가정한다 (frame_id += 1, 칼만 predict 1회).
실시간 처리에서 프레임을 k장 건너뛰면 실제로는 k+1 프레임만큼 이동했는데 예측은 1프레임만 앞서 있어
IoU 매칭이 빗나가고(추적 끊김 · 옆 물체로 id 넘어감), lost 트랙 보관 시간(max_time_lost)도 실제 시간보다 길어진다.

보정: 매 update 직전에 건너뛴 프레임 수 gap 을 알려 주면
  - frame_id 를 gap 만큼 미리 올리고 (lost 보관 시간이 실제 프레임 기준이 됨)
  - 칼만 예측을 1 + gap 번 돌린다 (속도·불확실성이 실제 경과 프레임만큼 진행)
tracker 인스턴스의 multi_predict 만 감싸므로 Ultralytics 코드는 고치지 않는다.

count/deploy/app/core.py 에 같은 코드가 있다 (배포 패키지는 저장소에 의존하지 않음). 고치면 둘 다 고칠 것.
"""

MAX_GAP = 60  # 이보다 길게 끊기면(재접속 등) 예측이 의미 없음 — 상한만 두고 그대로 진행 (트랙은 lost 처리됨)


def install(tracker):
    """tracker(BYTETracker/BOTSORT) 에 간격 보정을 한 번 설치. 이미 설치돼 있으면 그대로."""
    if getattr(tracker, "_gap_installed", False): return tracker
    orig = tracker.multi_predict
    tracker._gap = 0

    def multi_predict(tracks):
        n = 1 + tracker._gap
        tracker._gap = 0  # 이번 update 에서만 적용
        for _ in range(n): orig(tracks)

    tracker.multi_predict = multi_predict
    tracker._gap_installed = True
    return tracker


def set_gap(tracker, gap):
    """다음 update() 전에 호출. gap = 지난 처리 프레임과 이번 프레임 사이에 건너뛴 프레임 수."""
    gap = int(max(0, min(gap, MAX_GAP)))
    install(tracker)
    tracker._gap = gap
    tracker.frame_id += gap  # update() 가 +1 → 합계 gap+1 = 실제 경과 프레임


def tracker_of(model):
    """YOLO.track(persist=True) 이 만든 추적기 (첫 track 호출 전이면 None)."""
    p = getattr(model, "predictor", None)
    tr = getattr(p, "trackers", None) if p is not None else None
    return tr[0] if tr else None
