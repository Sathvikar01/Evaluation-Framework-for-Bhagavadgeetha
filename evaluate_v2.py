"""Backward-friendly delegator for the isolated Evaluation V2 CLI."""

from src.evaluation_v2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
