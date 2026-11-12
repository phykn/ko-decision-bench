# ko-decision-bench

A session continuing the rebuild uses groove to do the first unchecked task in `docs/groove/PLAN.md`.

- `src/kodb/__init__.py`: keeps Hugging Face downloads in `weights/` and the login token in its default place.
- `src/kodb/cli.py`: the `kodb` command (`build`, `run`, `score`, `list`).
- `src/kodb/tasks.py`: the seven tasks, building `data/<task>.jsonl`, and question helpers.
- `src/kodb/run.py`: runs one model on tasks and writes `results/<model>/<task>.json`.
- `src/kodb/score.py`: metrics, the Clean ranking, and the README tables.
- `src/kodb/adapters/`: one adapter per model family; `local.py` runs in this process, `server.py` starts a server and asks it over HTTP.
- `tests/`: `uv run --with pytest pytest`, a few seconds.

Results in `results/` were measured once; do not rewrite them. An adapter's prompt and readout must stay as they were when its results were measured.
