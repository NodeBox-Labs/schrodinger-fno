# Contributing to Schrödinger-FNO

We welcome contributions from the physics and machine learning communities! 

## How to Contribute
1. Fork the repository and create your branch from `main`.
2. Ensure you have installed the package in editable mode: `pip install -e .`
3. If you've added code that should be tested, add tests in the `tests/` directory.
4. Ensure the test suite passes locally: `pytest tests/ -v`
5. Format your code (we recommend `black` and `isort`).
6. Submit a Pull Request!

## Reporting Bugs
When reporting a bug, please include:
- Your OS and Python version.
- The exact command you ran.
- The full traceback.
- If possible, the potential \(V(x)\) and initial state \(\psi_0\) that caused the issue.