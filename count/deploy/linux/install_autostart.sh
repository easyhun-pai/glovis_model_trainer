#!/usr/bin/env bash
# 로그인하면(모니터 있는 데스크톱) 터미널 창과 함께 COUNT 를 자동 실행하도록 등록. 해제: ./linux/install_autostart.sh --remove
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"; F="$HOME/.config/autostart/count.desktop"
if [ "${1:-}" = "--remove" ]; then rm -f "$F"; echo "자동 실행 해제"; exit 0; fi
mkdir -p "$(dirname "$F")"
cat > "$F" <<EOT
[Desktop Entry]
Type=Application
Name=COUNT 차량 카운트
Exec=gnome-terminal --title=COUNT -- bash -c '"$DIR/linux/run_forever.sh"; exec bash'
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=20
EOT
echo "등록: $F"
echo "다음 로그인부터 자동 실행. 화면 잠금·절전은 [설정 > 전원]에서 끄는 것을 권장. 자동 로그인은 [설정 > 사용자]."
