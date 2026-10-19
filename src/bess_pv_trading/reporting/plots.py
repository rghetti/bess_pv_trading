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


def pv_power_allocation(frame: pd.DataFrame) -> pd.DataFrame:
    """Partition realized AC PV using PV-first charging at the shared bus.

    Uncurtailed PV supplies battery charging before grid imports are needed.
    Remaining PV is exported. Battery discharge and grid-supplied charging
    belong to other sources and are excluded from this PV-only partition.
    """
    pv = frame.pv_ac_mw.to_numpy(dtype=float)
    curtailed = frame.curtailment_mw.to_numpy(dtype=float)
    charge = frame.charge_mw.to_numpy(dtype=float)
    values = np.column_stack((pv, curtailed, charge))
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(curtailed > pv):
        raise ValueError("PV allocation requires finite nonnegative powers and curtailment no greater than PV")
    used_pv = pv - curtailed
    pv_to_battery = np.minimum(used_pv, charge)
    allocation = frame[[key for key in ("interval_id", "start", "end") if key in frame]].copy()
    allocation["pv_ac_mw"] = pv
    allocation["pv_to_grid_mw"] = used_pv - pv_to_battery
    allocation["pv_to_battery_mw"] = pv_to_battery
    allocation["pv_curtailed_mw"] = curtailed
    return allocation


def plot_pv_allocation(frame: pd.DataFrame, path: Path, *, timezone: str) -> pd.DataFrame:
    """Stack the three realized PV destinations and overlay their production total."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allocation = pv_power_allocation(frame)
    starts = pd.to_datetime(allocation.start, utc=True).dt.tz_convert(timezone)
    ends = pd.to_datetime(allocation.end, utc=True).dt.tz_convert(timezone)
    durations = (ends - starts).dt.total_seconds().to_numpy() / 3600
    fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    # Interval-width stacks preserve mixed resolutions, missing intervals, and
    # the final midnight boundary without attributing non-PV energy to PV.
    bottom = np.zeros(len(allocation))
    for column, label, color in (
            ("pv_to_grid_mw", "PV to grid", "C0"),
            ("pv_to_battery_mw", "PV to battery", "C2"),
            ("pv_curtailed_mw", "PV curtailed", "C1")):
        values = allocation[column].to_numpy()
        energy = float(np.dot(values, durations))
        axis.bar(starts, values, width=ends-starts, bottom=bottom,
            align="edge", linewidth=0, color=color, alpha=0.7,
            label=f"{label} ({energy:.2f} MWh)")
        bottom += values
    plot_interval_series(axis, starts, ends, allocation.pv_ac_mw,
        color="black", linewidth=1.7, label="Realized PV production")
    axis.set_title("Realized PV production: grid export, battery charging and curtailment")
    axis.set_ylabel("PV power (MW AC)")
    axis.set_xlabel(f"Delivery time ({timezone})")
    axis.set_xlim(starts.iloc[0], ends.iloc[-1])
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(ncol=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return allocation


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
