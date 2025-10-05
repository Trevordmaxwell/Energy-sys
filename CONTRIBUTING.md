# Contributing to UELM

## Getting Started
1. Fork the repository and clone your fork.
2. Run `make install` to set up the virtual environment and CPU dependencies.
3. Execute `make test` before submitting any changes.

## Coding Guidelines
- Follow PEP 8 and type annotate new Python code where practical.
- Keep CPU-first constraints in mind (avoid unnecessary allocations, prefer float32).
- Write unit tests for new functionality and update documentation when behavior changes.

## Pull Request Process
- Create feature branches from `main`.
- Ensure commits are logically grouped and descriptive.
- Run `make format` and `make test` locally.
- Fill out the PR template summarizing scope, tests, and hardware used.
