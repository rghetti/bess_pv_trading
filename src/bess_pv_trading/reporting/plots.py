"""Non-interactive plots labeled with the actual numerical experiment scope."""

from pathlib import Path
import numpy as np


def plot_dispatch(frame, prices, summary, path: Path) -> None:
    """Plot AC bus power, stored energy, commercial position, and cash flow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = np.arange(len(frame))
    fig, axes = plt.subplots(4,1,figsize=(12,11),sharex=True,layout="constrained")
    for name in ("pv_ac_mw","charge_mw","discharge_mw","grid_mw"):
        axes[0].plot(time,frame[name],label=name)
    axes[0].set_ylabel("AC MW"); axes[0].legend(ncol=4)
    axes[1].plot(time,frame.energy_start_mwh,label="internal battery energy")
    axes[1].set_ylabel("MWh"); axes[1].legend()
    axes[2].plot(time,frame.grid_mw*0.25,label="physical grid MWh")
    axes[2].plot(time,frame.net_position_mwh,label="accepted net position MWh")
    axes[2].set_ylabel("MWh"); axes[2].legend()
    for auction,values in prices.items():
        axes[3].plot(time,values,label=auction)
    axes[3].set_ylabel("EUR/MWh"); axes[3].legend(ncol=4)
    axes[3].set_xlabel("Quarter-hour interval in local delivery day")
    fig.suptitle(f"Synthetic {summary['strategy']}; realized profit EUR {summary['realized_operating_profit_eur']:.2f}")
    fig.savefig(path,dpi=120); plt.close(fig)
