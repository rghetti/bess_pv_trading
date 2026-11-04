"""Render presentation figures from the frozen participation evidence."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DESTINATION = Path(__file__).resolve().parent
PROJECT = DESTINATION.parents[2]
BLUE = "#24516c"
ORANGE = "#b06b24"


def arrow(axis, start, end, color=BLUE):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=1.3,
    ))


def draw_data_flow():
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.set(xlim=(0, 10), ylim=(0, 4.2))
    axis.axis("off")
    boxes = [
        (0.12, "Eligible weather", "Updated deterministic issue\n50 frozen weather members"),
        (3.48, "AC production paths", "Weather-to-power conversion\nDelivery grid and night policy"),
        (6.84, "Auction decisions", "Participation and battery plan\nSubmit the current order"),
    ]
    for horizontal, title, body in boxes:
        axis.add_patch(FancyBboxPatch(
            (horizontal, 1.52), 3.02, 1.48, boxstyle="round,pad=0.07",
            facecolor="#edf3f8", edgecolor=BLUE, linewidth=1.2,
        ))
        axis.text(horizontal + 1.51, 2.65, title, ha="center", va="center",
                  fontsize=11, weight="bold", color=BLUE)
        axis.text(horizontal + 1.51, 2.07, body, ha="center", va="center",
                  fontsize=9.4, linespacing=1.65)
    for horizontal in [3.18, 6.54]:
        arrow(axis, (horizontal, 2.25), (horizontal + 0.24, 2.25))

    axis.add_patch(FancyBboxPatch(
        (3.48, 3.47), 3.02, 0.55, boxstyle="round,pad=0.06",
        facecolor="#f6f3ed", edgecolor=ORANGE, linewidth=1.1,
    ))
    axis.text(4.99, 3.75, "Historical power and irradiance\nFarm metadata",
              ha="center", va="center", fontsize=9, linespacing=1.3)
    arrow(axis, (4.99, 3.38), (4.99, 3.09), ORANGE)
    axis.text(8.33, 0.95, "Physical delivery and settlement",
              ha="center", fontsize=9.5, color=BLUE)
    arrow(axis, (8.35, 1.42), (8.35, 1.13))
    axis.plot([8.35, 8.35, 1.63], [0.82, 0.35, 0.35],
              color=ORANGE, linewidth=1.3)
    arrow(axis, (1.63, 0.35), (1.63, 1.42), ORANGE)
    axis.text(5, 0.52, "New forecasts, actual stored energy and fixed commitments",
              ha="center", fontsize=9.4, color="#81501d")
    figure.savefig(DESTINATION / "data_to_decisions.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def draw_participation():
    source = PROJECT / "docs/report_evidence/2026-11-04/continuous_participation.csv"
    with source.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["farm"] == "1"]
    labels = [row["profile"].replace("sequential_all_auctions", "Sequential all auctions")
              .replace("perfect_pv_all_auctions", "Perfect production") for row in rows]
    deterministic = [float(row["net_profit_eur_deterministic"]) / 1000 for row in rows]
    ensemble = [float(row["net_profit_eur_ensemble"]) / 1000 for row in rows]
    vertical = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(8.6, 5.1), layout="constrained")
    axis.barh([value - 0.18 for value in vertical], deterministic, height=0.32,
              color=BLUE, label="Updating deterministic", zorder=3)
    axis.barh([value + 0.18 for value in vertical], ensemble, height=0.32,
              color=ORANGE, label="Strict 51-path planning", zorder=3)
    axis.set_yticks(vertical, labels)
    axis.invert_yaxis()
    axis.set_xlabel("January operating profit (thousand EUR)")
    axis.set_xlim(0, 11.5)
    axis.set_xticks([0, 2, 4, 6, 8, 10])
    axis.grid(axis="x", color="#dce3e8", linewidth=0.7)
    axis.axhline(14.5, color="#8a949b", linewidth=0.8)
    axis.tick_params(axis="both", labelsize=9)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2,
                fontsize=9, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(DESTINATION / "participation_profiles.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


if __name__ == "__main__":
    draw_data_flow()
    draw_participation()
