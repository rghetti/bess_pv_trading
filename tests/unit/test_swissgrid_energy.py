"""Swissgrid timestamp shifts, energy units and self-contained snapshot behavior."""

from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.swissgrid_energy import acquire_energy_statistics, parse_energy_statistics


ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def settings_and_source(year=2025, label="interval_start"):
    """Use the production schema with a small subset of fields for isolated examples."""
    settings = load_config(ROOT / "configs/api_experiments/swissgrid_energy_history.yaml")["swissgrid_energy"]
    names = ["generation", "consumption", "physical_ch_to_it", "physical_it_to_ch", "afrr_positive_price"]
    settings["fields"] = {name: settings["fields"][name] for name in names}
    settings["net_borders"] = {"it": settings["net_borders"]["it"]}
    settings["plots"] = False
    source = {"year": year, "timestamp_label": label, "require_complete_year": False,
              "filename": f"EnergieUebersichtCH-{year}.xlsx", "url": "https://www.swissgrid.ch/example.xlsx"}
    return settings, source


def workbook_bytes(settings, timestamps, *, missing_import=False, wrong_unit=False):
    """Construct a minimal XLSX using native inline strings and cached formula values."""
    worksheet = ET.Element(f"{{{NAMESPACE}}}worksheet")
    data = ET.SubElement(worksheet, f"{{{NAMESPACE}}}sheetData")
    header = {field["column"]: field["expected_header"] for field in settings["fields"].values()}
    units = {field["column"]: field["native_unit"] for field in settings["fields"].values()}
    if wrong_unit:
        units["C"] = "MWh"
    values = {"C": "250000", "D": "500000", "Q": "125000", "R": "25000", "V": "-20"}
    rows = [header, units, *({"A": timestamp, **values} for timestamp in timestamps)]
    for row_number, contents in enumerate(rows, start=1):
        # Omitted XLSX cells become a per-row null when other rows contain the same column.
        if missing_import and row_number == 3:
            contents = {column: value for column, value in contents.items() if column != "R"}
        row = ET.SubElement(data, f"{{{NAMESPACE}}}row", r=str(row_number))
        for column, value in contents.items():
            cell = ET.SubElement(row, f"{{{NAMESPACE}}}c", r=f"{column}{row_number}")
            if row_number <= 2 or (column == "A" and "." in value and " " in value):
                cell.set("t", "inlineStr")
                inline = ET.SubElement(cell, f"{{{NAMESPACE}}}is")
                ET.SubElement(inline, f"{{{NAMESPACE}}}t").text = value
            else:
                if column == "A":
                    ET.SubElement(cell, f"{{{NAMESPACE}}}f").text = "Datetime!A3"
                ET.SubElement(cell, f"{{{NAMESPACE}}}v").text = value
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="' + NAMESPACE +
                         '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                         '<sheets><sheet name="Zeitreihen0h15" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(worksheet))
    return output.getvalue()


@pytest.mark.parametrize("year,label,timestamp", [
    (2024, "interval_end", "01.01.2024 00:15"),
    (2025, "interval_start", "01.01.2025 00:00"),
    (2026, "interval_start", "46023"),
])
def test_timestamp_contract_and_units(year, label, timestamp):
    """All three provider timestamp representations align to the same local interval."""
    settings, source = settings_and_source(year, label)
    frame, metadata = parse_energy_statistics(workbook_bytes(settings, [timestamp]), settings, source)
    expected = pd.Timestamp(f"{year}-01-01", tz="Europe/Zurich").tz_convert("UTC")
    assert frame.product_start.iloc[0] == expected
    assert frame.product_end.iloc[0] == expected + pd.Timedelta(minutes=15)
    assert frame.generation_mw.iloc[0] == 1000
    assert frame.physical_net_export_it_mw.iloc[0] == 400
    assert frame.afrr_positive_price_eur_mwh.iloc[0] == -20
    assert metadata["energy_to_mean_mw_divisor"] == 250


def test_end_labels_preserve_both_autumn_folds():
    """Localizing before subtracting the interval avoids a DST fold shift."""
    settings, source = settings_and_source(2024, "interval_end")
    times = [f"27.10.2024 02:{minute:02d}" for minute in (0, 15, 30, 45)] * 2
    frame, _ = parse_energy_statistics(workbook_bytes(settings, times), settings, source)
    assert not frame.product_start.duplicated().any()
    assert frame.product_start.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert len(frame) == 8


def test_missing_flows_stay_missing_and_schema_drift_fails():
    """An absent directional leg cannot silently become a zero net flow."""
    settings, source = settings_and_source()
    frame, _ = parse_energy_statistics(
        workbook_bytes(settings, ["01.01.2025 00:00", "01.01.2025 00:15"], missing_import=True), settings, source
    )
    assert np.isnan(frame.physical_net_export_it_mw.iloc[0])
    assert frame.physical_net_export_it_mw.iloc[1] == 400
    with pytest.raises(ValueError, match="header/unit"):
        parse_energy_statistics(workbook_bytes(settings, ["01.01.2025 00:00"], wrong_unit=True), settings, source)
    source["require_complete_year"] = True
    with pytest.raises(ValueError, match="missing.*interval timestamps"):
        parse_energy_statistics(workbook_bytes(settings, ["01.01.2025 00:00"]), settings, source)


def test_database_remains_self_contained_without_source_workbook(tmp_path):
    """A fresh machine can read the cached snapshot after the external XLSX is removed."""
    settings, source = settings_and_source()
    settings.update(database_path=str(tmp_path / "market.duckdb"), database_threads=1,
                    raw_directory=str(tmp_path), report_path=str(tmp_path / "report.json"), sources=[source])
    payload = workbook_bytes(settings, ["01.01.2025 00:00"])
    workbook = tmp_path / source["filename"]
    workbook.write_bytes(payload)
    first = acquire_energy_statistics({"swissgrid_energy": settings}, offline=True)
    workbook.unlink()
    second = acquire_energy_statistics({"swissgrid_energy": settings}, offline=True)
    assert first["status"] == second["status"] == "complete"
    assert second["sources"][0]["status"] == "cached"
    manifest = second["sources"][0]["manifest"]
    store = DuckDBCache(settings["database_path"], threads=1)
    assert store.load_raw(manifest["request"]) == payload
    assert store.load(manifest["request"])[0].physical_net_export_it_mw.iloc[0] == 400


def test_online_refresh_fetches_even_when_a_local_workbook_exists(tmp_path, monkeypatch):
    """A deliberate refresh must not keep reimporting an old local download."""
    settings, source = settings_and_source()
    settings.update(database_path=str(tmp_path / "market.duckdb"), database_threads=1,
                    raw_directory=str(tmp_path), report_path=str(tmp_path / "report.json"), sources=[source])
    payload = workbook_bytes(settings, ["01.01.2025 00:00"])
    (tmp_path / source["filename"]).write_bytes(payload)
    calls = []

    class Transport:
        """Track the authorized public fetch without issuing a network request."""

        def get(self, specification):
            calls.append(specification.url)
            return payload

    monkeypatch.setattr("bess_pv_trading.data.swissgrid_energy.SecureTransport", lambda **kwargs: Transport())
    report = acquire_energy_statistics({"swissgrid_energy": settings}, refresh=True)
    assert report["status"] == "complete"
    assert calls == [source["url"]]
