"""기준선 설정 화면: 현재 카메라 화면 위에 통과 기준선을 마우스로 긋고, 야드 쪽을 지정한다.

- 저장된 선이 있으면 먼저 보여 주고, 아무 키도 안 누르면 timeout 초 뒤 그대로 시작 (무인 재시작 대비)
- Enter/Space: 이 선으로 시작 · R: 다시 그리기 · Esc: 종료
- 그리기: 선의 두 끝을 차례로 클릭 → 야드 쪽(선적 차량이 출발하는 쪽)을 한 번 클릭
- 끝점을 화면 왼쪽/오른쪽 가장자리 가까이(2% 안) 찍으면 선을 화면 밖까지 연장한다.
  (카메라에 가까운 차가 선을 넘기 전에 화면 밖으로 잘려 나가는 화각에서 가장자리 통과도 세기 위함)
"""
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIN = "COUNT - 기준선 설정"


FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",                       # Windows
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu fonts-nanum
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # fonts-noto-cjk
)
_FONT_PATH = None


def _font_path():
    """한글이 되는 폰트 파일. 후보 경로 → fc-match(리눅스) 순. 없으면 None (한글이 깨진다 → install.sh 가 fonts-nanum 설치)."""
    global _FONT_PATH
    if _FONT_PATH is not None: return _FONT_PATH or None
    import os, shutil, subprocess
    path = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
    if path is None and shutil.which("fc-match"):
        try:
            p = subprocess.run(["fc-match", "-f", "%{file}", "sans:lang=ko"], capture_output=True, text=True, timeout=5).stdout.strip()
            path = p if p and os.path.exists(p) else None
        except Exception: pass
    _FONT_PATH = path or ""
    return path


def _font(size):
    p = _font_path()
    if p:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()


def cross_raw(A, B, q):
    return (B[0] - A[0]) * (q[1] - A[1]) - (B[1] - A[1]) * (q[0] - A[0])


def extend_edges(A, B, W, margin=0.02, out=0.1):
    """가장자리 근처 끝점을 선 방향 그대로 화면 밖(가로 out 비율)까지 늘린다."""
    A, B = np.array(A, float), np.array(B, float); d = B - A
    if abs(d[0]) < 1e-6: return A, B
    for P, sgn in ((A, -1), (B, 1)):
        if P[0] <= margin * W: tx = -out * W
        elif P[0] >= (1 - margin) * W: tx = (1 + out) * W
        else: continue
        P[:] = P + d * ((tx - P[0]) / d[0])
    return A, B


def setup_line(frame, saved=None, timeout_s=15):
    """saved: {"line": [[x,y],[x,y]], "yard_side": ±1} 또는 None. 반환: 같은 형식 dict, Esc 면 None."""
    H, W = frame.shape[:2]
    s = min(1600 / W, 900 / H, 1.0); dw, dh = int(W * s), int(H * s)
    base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    f_big, f_sm = _font(24), _font(18)
    st = dict(mode="confirm" if saved else "p1", pts=[], line=None, yard=None, mouse=None, t0=time.time())
    if saved: st["line"] = [np.array(p, float) for p in saved["line"]]; st["yard"] = saved["yard_side"]

    def on_mouse(ev, x, y, *_):
        st["mouse"] = (x / s, y / s)
        if ev != cv2.EVENT_LBUTTONDOWN: return
        q = np.array([x / s, y / s])
        if st["mode"] == "p1": st["pts"] = [q]; st["mode"] = "p2"
        elif st["mode"] == "p2":
            if np.linalg.norm(q - st["pts"][0]) < 20 / s: return
            A, B = st["pts"][0], q
            if A[0] > B[0]: A, B = B, A
            st["line"] = list(extend_edges(A, B, W)); st["mode"] = "yard"
        elif st["mode"] == "yard":
            c = cross_raw(*st["line"], q)
            if abs(c) < 1e-6: return
            st["yard"] = 1 if c > 0 else -1; st["mode"] = "check"

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback(WIN, on_mouse)
    while True:
        im = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im, "RGBA")
        m = st["mode"]
        if m == "p2" and st["mouse"] is not None:
            a = st["pts"][0] * s; b = np.array(st["mouse"]) * s
            d.line([tuple(a), tuple(b)], fill=(0, 230, 255, 200), width=3); d.ellipse([a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5], fill=(0, 230, 255))
        if st["line"] is not None:
            A, B = st["line"][0] * s, st["line"][1] * s
            d.line([tuple(A), tuple(B)], fill=(0, 230, 255, 255), width=4)
            if st["yard"] is not None:
                mid = (A + B) / 2; n = np.array([-(B - A)[1], (B - A)[0]]); n /= np.linalg.norm(n) + 1e-9
                if np.sign(cross_raw(A, B, mid + n)) != st["yard"]: n = -n
                for lab, off, col in (("야드 쪽 (선적 출발)", n * 60, (40, 200, 90)), ("배 쪽 (하역 출발)", -n * 60, (255, 150, 60))):
                    p = mid + off; tw = d.textlength(lab, font=f_sm)
                    d.rounded_rectangle([p[0] - tw / 2 - 8, p[1] - 14, p[0] + tw / 2 + 8, p[1] + 14], 6, fill=col + (220,))
                    d.text((p[0] - tw / 2, p[1] - 12), lab, font=f_sm, fill=(255, 255, 255))
        msg = {"p1": "기준선 시작점을 클릭하세요 (배수로 등 차량이 넘는 선을 따라)",
               "p2": "기준선 끝점을 클릭하세요",
               "yard": "야드 쪽(선적 차량이 출발하는 쪽)을 한 번 클릭하세요",
               "check": "Enter: 이 선으로 시작   R: 다시 그리기   Esc: 종료"}.get(m)
        if m == "confirm":
            left = max(0, timeout_s - (time.time() - st["t0"]))
            msg = f"저장된 기준선입니다.  Enter: 시작   R: 다시 그리기   Esc: 종료   ({left:.0f}초 뒤 자동 시작)"
            if left <= 0: break
        tw = d.textlength(msg, font=f_big)
        y0 = dh - 54  # 아래쪽 (위쪽은 카메라 날짜·시각 표시)
        d.rounded_rectangle([10, y0, tw + 34, y0 + 42], 8, fill=(15, 20, 26, 215)); d.text((22, y0 + 6), msg, font=f_big, fill=(255, 255, 255))
        cv2.imshow(WIN, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: k = 27
        if k == 27: cv2.destroyWindow(WIN); return None
        if k in (ord("r"), ord("R")): st.update(mode="p1", pts=[], line=None, yard=None)
        if k in (13, 32) and m in ("confirm", "check"): break
    cv2.destroyWindow(WIN)
    return dict(line=[[round(float(v), 1) for v in p] for p in st["line"]], yard_side=int(st["yard"]), frame_size=[W, H])
