"""Exercise full-year source selection without opening a market database."""

from pathlib import Path

import pandas as pd

from scripts.prepare_full_year_price_inputs import prepare_merged_inputs


def test_merged_inputs_retain_fallback_snapshots_and_source_priority(tmp_path):
    """GME and consolidated RTE win overlaps, while missing intervals use fallbacks."""
    def rows(provider, dataset, starts, values, zone):
        starts = pd.to_datetime(starts, utc=True)
        return pd.DataFrame({"provider": provider, "dataset": dataset, "zone": zone,
            "product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
            "value": values, "_snapshot_id": provider + dataset, "_row_id": range(len(starts))})

    first = "2026-10-01T00:00:00Z"
    second = "2026-10-01T00:15:00Z"
    sources = {"nord_auctions": rows("gme", "day_ahead_prices", [first], [10.0], "NORD"),
        "energy_charts_nord_da": rows("energy_charts", "day_ahead_prices", [first, second], [10.0, 11.0], "NORD"),
        "france_balance": rows("rte", "electricity_balance", [first], [20.0], "FR"),
        "france_balance_realtime": rows("rte", "electricity_balance_realtime", [first, second], [21.0, 22.0], "FR")}

    def source_builder(config):
        raw = Path(config["price_search_inputs"]["output_directory"])
        raw.mkdir(parents=True)
        table_info = {}
        for name, frame in sources.items():
            path = raw / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            table_info[name] = {"path": path.name, "sha256": "source-hash", "selected_row_sha256": "row-hash",
                                "snapshot_sources": [{"snapshot_id": name}]}
        (raw / "manifest.json").write_text("{}")
        return {"status": "complete", "tables": table_info, "selection_sha256": "selection"}

    output = tmp_path / "merged"
    report = prepare_merged_inputs({"price_search_inputs": {}}, tmp_path / "raw", output,
                                   source_builder=source_builder)
    da = pd.read_parquet(output / "nord_auctions.parquet")
    france = pd.read_parquet(output / "france_balance.parquet")
    assert da.provider.tolist() == ["gme", "energy_charts"]
    assert da.value.tolist() == [10.0, 11.0]
    assert france.value.tolist() == [20.0, 22.0]
    assert report["tables"]["nord_auctions"]["fallback_snapshot_sources"] == [{"snapshot_id": "energy_charts_nord_da"}]
    assert report["fallback_policy"]["france"]["max_overlap_absolute_difference"] == 1.0
