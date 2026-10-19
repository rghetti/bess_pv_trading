"""Non-interactive plots labeled with the actual numerical experiment scope."""

from pathlib import Path
import numpy as np
import pandas as pd


def _plot_interval_boundaries(axis, starts, ends, start_values, end_values, **kwargs):
    """Draw both actual boundaries, retaining gaps and timezone-aware timestamps."""
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    left, right = np.asarray(start_values, dtype=float), np.asarray(end_values, dtype=float)
    if not (len(starts) == len(ends) == len(left) == len(right)):
        raise ValueError("Plot interval boundaries and values must have equal lengths")
    if not len(starts):
        return []
    if np.any(ends <= starts):
        raise ValueError("Plot intervals must have positive duration")
    times = np.column_stack((starts.to_pydatetime(), ends.to_pydatetime())).ravel()
    values = np.column_stack((left, right)).ravel()
    # Break the line across missing intervals rather than inventing their values.
    gaps = 2 * (np.flatnonzero(ends[:-1] != starts[1:]) + 1)
    return axis.plot(np.insert(times, gaps, times[gaps]),
                     np.insert(values, gaps, np.nan), **kwargs)


def plot_interval_series(axis, starts, ends, values, **kwargs):
    """Hold interval power/price through its end, including the delivery-day boundary."""
    return _plot_interval_boundaries(axis, starts, ends, values, values, **kwargs)


def plot_boundary_series(axis, starts, ends, start_values, end_values, **kwargs):
    """Plot state or accumulated cash at every known start and end boundary."""
    return _plot_interval_boundaries(axis, starts, ends, start_values, end_values, **kwargs)


def plot_dispatch(frame, prices, summary, path: Path, *, timezone: str) -> None:
    """Plot AC bus power, stored energy, commercial position, and cash flow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    starts = pd.to_datetime(frame.start, utc=True).dt.tz_convert(timezone)
    ends = pd.to_datetime(frame.end, utc=True).dt.tz_convert(timezone)
    durations = (ends - starts).dt.total_seconds().to_numpy() / 3600
    fig, axes = plt.subplots(4,1,figsize=(12,11),sharex=True,layout="constrained")
    for name in ("pv_ac_mw","charge_mw","discharge_mw","grid_mw"):
        plot_interval_series(axes[0], starts, ends, frame[name], label=name)
    axes[0].set_ylabel("AC MW"); axes[0].legend(ncol=4)
    plot_boundary_series(axes[1], starts, ends, frame.energy_start_mwh,
                         frame.energy_end_mwh, label="internal battery energy")
    axes[1].set_ylabel("MWh"); axes[1].legend()
    plot_interval_series(axes[2], starts, ends, frame.grid_mw * durations, label="physical grid MWh")
    plot_interval_series(axes[2], starts, ends, frame.net_position_mwh, label="accepted net position MWh")
    axes[2].set_ylabel("MWh"); axes[2].legend()
    for auction,values in prices.items():
        plot_interval_series(axes[3], starts, ends, values, label=auction)
    axes[3].set_ylabel("EUR/MWh"); axes[3].legend(ncol=4)
    axes[3].set_xlabel(f"Delivery time ({timezone}; includes final interval end)")
    axes[0].set_xlim(starts.iloc[0], ends.iloc[-1])
    fig.suptitle(f"{summary['strategy']}; realized profit EUR {summary['realized_operating_profit_eur']:.2f}")
    fig.savefig(path,dpi=120); plt.close(fig)
