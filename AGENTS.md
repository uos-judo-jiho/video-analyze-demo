# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python video-analysis MVP. Core logic lives in [judo_highlight_mvp.py](C:/Users/itman/dev/uos-judo-jiho/video-analyze-demo/judo_highlight_mvp.py), which loads a YOLO pose model, detects fall-like events, and cuts highlight clips. [main.py](C:/Users/itman/dev/uos-judo-jiho/video-analyze-demo/main.py) is a minimal stub entry point. Sample inputs live under `videos/`, generated clips are written to `outputs/`, and dependency metadata is in `pyproject.toml` and `uv.lock`.

## Build, Test, and Development Commands
Use `uv` with Python 3.12+.

- `uv sync` installs project dependencies into `.venv`.
- `uv run python judo_highlight_mvp.py videos/sample-1.mp4` runs the analyzer against a sample video.
- `uv run python judo_highlight_mvp.py videos/sample-1.mp4 --output outputs --sample-fps 5 --before 10 --after 10` runs with explicit tuning parameters.
- `uv run python main.py` runs the placeholder entry point.

`ffmpeg` must be installed and available on `PATH`; clip cutting will fail without it.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, type hints, and small focused functions. Use `snake_case` for variables, functions, and file names; use `PascalCase` for dataclasses such as `FallEvent`. Keep argument parsing in `main()` and put reusable processing logic in standalone functions. Prefer clear numeric parameter names like `sample_fps`, `before_sec`, and `after_sec`.

## Testing Guidelines
There is no test suite yet. Add new tests under `tests/` using `pytest`, with file names like `test_merge_events.py`. Cover pure logic first: event merging, angle calculation, and fall scoring. Run tests with `uv run pytest` once tests are added. Avoid relying on large video fixtures unless the behavior cannot be validated with small synthetic inputs.

## Commit & Pull Request Guidelines
Current history uses short imperative subjects (`init`). Keep commits concise and action-oriented, for example `add clip export bounds check` or `fix fall scoring threshold`. Pull requests should include a short description, the sample video or scenario used for validation, and representative output paths or screenshots if clip output changed.

## Generated Assets & Data
Do not commit `.venv/`, Python build artifacts, or ad hoc output clips. Treat `videos/` as local sample data unless the PR explicitly updates fixtures needed for reproducible testing.
