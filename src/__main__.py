"""Run the application using python -m bess_pv_trading."""

from .cli import main

# Reject invalid input when `__name__ == '__main__'`.
if __name__ == "__main__":
    raise SystemExit(main())
