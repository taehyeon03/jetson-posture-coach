# Jetson 온디바이스 웹 MVP

카메라 캡처, TensorRT 추론, 개인 기준 자세 판정, JPEG 인코딩, HTTP 서버가 모두 Jetson Nano에서 실행된다. 브라우저는 실시간 영상과 관절, 현재 판정, 변화량, 보정 상태와 지속 경고 이력을 표시한다.

## 실행

Jetson의 저장소 루트에서:

```bash
./scripts/nano_python.sh web_mvp/server.py \
  --engine models/pose-resnet18-fp16.engine \
  --host 0.0.0.0 --port 8080
```

같은 Tailscale 네트워크의 개발 PC 브라우저에서 `http://JETSON_TAILSCALE_IP:8080`을 연다. SSH 포트 포워딩은 개발용 우회 수단일 뿐이며, 영상·추론 연산은 Jetson에서 수행한다.

## 엔드포인트

- `/` — 최소 웹 화면
- `/video.mjpeg` — 최신 프레임만 보내는 MJPEG 스트림
- `/api/pose` — 최신 관절 좌표, 자세 판정, 변화량, FPS, 추론 지연과 경고 이력
- `/api/health` — 프레임 수신 여부
- `POST /api/calibrate` — 3초 준비 후 5초간 개인의 바른 자세 기준 수집

이 MVP는 지연 누적을 막기 위해 카메라와 스트림 모두 오래된 프레임을 버린다. MJPEG는 빠른 검증용이며, 추후 영상 전송만 WebRTC로 교체할 수 있다.
