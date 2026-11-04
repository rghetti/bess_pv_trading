# Ensemble coverage and annual-study readiness

The coordinate inventory on 4 November 2026 finds **62 candidate delivery dates** shared by the archived weather ensembles and the four farms' measurement timestamps: **1 January through 1 February (32 days)** and **2–31 August (30 days)**. The January archive includes the 31 January issue cycle, so **1 February is one extra candidate** beyond the previous January-only study window. Its numerical weather values and PV conversion still require ordinary preflight.

These are 62 common calendar dates, not 248 independent farm observations. Coordinate coverage does not establish finite measured power, acceptable QC, calibrated forecasts or model availability. Missing measurement values retain the configured reporting and fallback policy during full preflight.

| Archive | Stored cycles | Leads | Members | Coordinate-supported delivery dates with measurement overlap |
|---|---:|---:|---:|---|
| `nwp_ens_20260101_20260131.zarr` | 32 | 49 | 50 | 2026-01-01 through 2026-02-01 |
| `nwp_ens_20260801_20260930.zarr` | 61 | 49 | 50 | 2026-08-02 through 2026-08-31 |

The main measurement archive exposes 58,368 quarter-hour timestamps covering January 2025 through August 2026. September weather does not create September measurement coverage. No missing months or cycles are fabricated. The first August delivery date has no preceding 31 July ensemble cycle in the available archives.

The inventory reads coordinate arrays and variable shapes, checks `valid_time = cycle_time + lead_time`, and verifies quarter-hour midpoint and hourly endpoint coverage on the local delivery grid. DST days therefore contain their actual 92 or 100 quarter-hours. It uses the configured six-hour weather availability assumption and the first auction decision cutoff. Retrieval dates are not treated as publication evidence.

`src/data/ensemble_inventory.py` selects archives by stored issue cycles and requested farms. Member counts come from each archive; neither 50 members nor the January/August filenames are required. Multiple archives can support a full-year study as additional data arrive. Duplicate issue cycles fail unless an explicit `first` or `last` source-precedence policy is selected, and the inventory retains all overlapping candidate paths.

Run the metadata-only inventory with:

```bash
python scripts/inventory_energy_study_data.py --config configs/research/ensemble_coverage_2026.yaml
```

When a full-year ensemble arrives, append its real path. If it overlaps the existing archives, explicitly select the intended precedence:

```bash
python scripts/inventory_energy_study_data.py --source data/farms/full_2026_ensemble.zarr --overlap-policy last
```

That filename is an invocation example, not an assertion that the file exists. A missing path is recorded as missing. Use `--measurement-source` when a newer measured-power archive becomes available. Outputs are grouped under `outputs/research_inventory/2026`: a complete date-level CSV, JSON evidence and Markdown inventory. Metadata and coordinate checksums are explicitly distinguished from full payload checksums; ordinary replay preflight separately hashes the converted scenario values actually consumed.

The shared calendar currently ends on 31 August 2026. The annual inventory configuration alone extends its validity to 31 December, as an explicit assumption that the configured gates remain unchanged. GME's [DTF03 revision 14](https://www.mercatoelettrico.org/Portals/0/Documents/it-IT/20251209DTF03rev14MPE.pdf), activated by the [January 2026 notice](https://gme.mercatoelettrico.org/it-it/Home/AvvisieComunicati/AvvisieComunicatiME?id=7282), is the current timing reference. Future calendar changes require updating the study configuration. The inventory does not authorize or launch an annual historical experiment.
