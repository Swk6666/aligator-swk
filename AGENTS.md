# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++ sources grouped in `core/`, `solvers/`, `modelling/`, `utils/`, `gar/`.
- `include/aligator/`: Public headers for the C++ API.
- `bindings/python/`: Python bindings and package `aligator` built from C++.
- `tests/`: C++ tests; `tests/python/`: pytest-based tests integrated with CTest.
- `examples/`, `bench/`, `doc/`, `assets/`: Usage examples, benchmarks, docs, assets.
- Build and env: `CMakeLists.txt`, `pixi.toml` tasks and feature envs.

## Build, Test, and Development Commands
- Quick start (recommended):
  - `pixi run build`: Configure and build to `build/`.
  - `pixi run test`: Run CTest (includes C++ and Python tests).
  - `pixi shell`: Enter the dev env to run `cmake`, `ctest`, `pytest` manually.
- Manual CMake/Ninja:
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build -j`
  - `ctest --test-dir build --output-on-failure`
- Clean: `pixi run clean`. Enable features via envs, e.g. `pixi run -e pinocchio build`.

## Coding Style & Naming Conventions
- C++: LLVM style via `clang-format` (`.clang-format`). Run `pre-commit` locally.
- CMake: formatted with `gersemi` (via pre-commit).
- Python: `ruff` lint + format (see `pyproject.toml`).
- Naming: C++ lives under namespace `aligator`; headers use `.hpp`. Prefer `UpperCamelCase` for types, `lower_snake_case` for functions/variables consistent with existing code.

## Testing Guidelines
- Frameworks: C++ tests use CTest; Python tests use `pytest` under `tests/python/` and are invoked by CTest.
- Run all: `pixi run test`. Filter: `ctest -R <pattern>`.
- Python-only (inside `pixi shell` after build): `pytest tests/python -q`.
- Add focused tests near related modules; keep current tests passing. Use `@pytest.mark.skipif` for optional features.

## Commit & Pull Request Guidelines
- Commit messages: concise, imperative; optional scope prefix (e.g., `tests:`, `solvers:`). Reference issues (`Fixes #123`).
- Before opening a PR:
  - Build succeeds: `pixi run build`.
  - Tests pass: `pixi run test`.
  - Linters clean: `pixi run -e lint lint` or `pre-commit run --all`.
- PRs should include a clear description, motivation, linked issues, and screenshots/plots for example updates.

## Security & Configuration Tips
- Do not commit `build/` outputs or large binaries.
- When using Conda, export `CMAKE_PREFIX_PATH=$CONDA_PREFIX` for manual builds.
- Optional features: `pinocchio`, `crocoddyl`, `openmp`, `cholmod` via Pixi envs.

