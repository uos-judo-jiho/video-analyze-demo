# Judo Highlight MVP

로컬 mp4 유도 영상을 분석해서 넘어짐처럼 보이는 시점을 찾고, 각 이벤트 기준 앞뒤 10초 하이라이트 클립을 `outputs/`에 생성하는 Python MVP입니다.

## Requirements

- Windows
- Python 3.12+
- `uv`
- `ffmpeg` on `PATH`

확인:

```powershell
ffmpeg -version
uv --version
```

## Install

```powershell
uv sync
```

모델 파일 `yolov8n-pose.pt`는 프로젝트 루트에 있어야 합니다.

## Run

기본 실행:

```powershell
uv run python judo_highlight_mvp.py videos/sample-1.mp4 --output outputs
```

옵션 포함 실행:

```powershell
uv run python judo_highlight_mvp.py videos/sample-1.mp4 --output outputs --sample-fps 5 --before 10 --after 10
```

## Output

실행 후 아래 파일들이 생성됩니다.

- `outputs/highlight_01_x_y.mp4`
- `outputs/highlight_02_x_y.mp4`
- `outputs/events.json`

`events.json`에는 병합된 이벤트의 `time_sec`, `confidence`, `aspect_ratio`, `body_angle_deg`, `start_sec`, `end_sec`가 저장됩니다.

## Detection Logic

초기 MVP는 `ultralytics`의 `YOLO("yolov8n-pose.pt")`를 사용합니다.

- 사람 bbox의 `width / height`
- 어깨/엉덩이 keypoint 기반 몸통 각도
- 두 조건을 점수화해 fall-like 여부 계산
- 가까운 timestamp는 8초 기준으로 병합

콘솔 로그에는 사람별 `aspect_ratio`, `body_angle_deg`, `confidence`, `fall_like`가 출력되므로 오탐 튜닝에 바로 사용할 수 있습니다.
