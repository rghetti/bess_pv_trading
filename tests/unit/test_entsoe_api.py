"""Source-syntax synthetic XML tests, never dependent on API tokens or network."""
import pandas as pd
import pytest

from bess_pv_trading.data.entsoe import (
    DATASETS, ENTSOE_ENDPOINT, NORD_EIC, EntsoeAPIError, EntsoeNoDataError,
    build_request, dataset_metadata, parse_entsoe_xml,
)


RETRIEVED_AT = "2026-10-16T12:00:00Z"


def period_xml(start, end, points, resolution="PT15M", value_field="price.amount"):
    """Build synthetic source-syntax periods, including explicit depublication."""
    point_tags = "".join(f"<Point><position>{position}</position>" +
                         (f"<{value_field}>{value}</{value_field}>" if value is not None else "") +
                         "</Point>" for position, value in points)
    return (f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
            f"<resolution>{resolution}</resolution>{point_tags}</Period>")


def document_xml(*, dataset="day_ahead_prices", curve="A03", periods=None,
                 revision="3", zone_eic=NORD_EIC):
    """Use the official Publication/GL tag vocabulary with invented test values."""
    specification = DATASETS[dataset]
    price = dataset == "day_ahead_prices"
    root = "Publication_MarketDocument" if price else "GL_MarketDocument"
    process = "" if price else "<process.processType>A01</process.processType>"
    domains = (f"<in_Domain.mRID>{zone_eic}</in_Domain.mRID><out_Domain.mRID>{zone_eic}</out_Domain.mRID>" if price else
               f"<{'out' if dataset == 'load_forecast' else 'in'}BiddingZone_Domain.mRID>{zone_eic}"
               f"</{'out' if dataset == 'load_forecast' else 'in'}BiddingZone_Domain.mRID>")
    units = ("<currency_Unit.name>EUR</currency_Unit.name><price_Measure_Unit.name>MWH</price_Measure_Unit.name>" if price else
             "<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>")
    psr = f"<MktPSRType><psrType>{specification['psr_type']}</psrType></MktPSRType>" if "psr_type" in specification else ""
    if periods is None:
        periods = period_xml("2025-01-01T00:00Z", "2025-01-01T01:00Z", [(1, -7), (3, 100)],
                             value_field="price.amount" if price else "quantity")
    return (f'<{root} xmlns="urn:synthetic:entsoe-schema-test"><mRID>invented-doc</mRID>'
            f"<revisionNumber>{revision}</revisionNumber><type>{specification['document_type']}</type>"
            f"{process}<createdDateTime>2025-01-02T10:00Z</createdDateTime>"
            f"<TimeSeries><mRID>invented-series</mRID>{domains}{units}{psr}"
            f"<curveType>{curve}</curveType>{periods}</TimeSeries></{root}>")


def parse(xml, dataset="day_ahead_prices"):
    """Keep the retrieval time explicit in every parser test."""
    return parse_entsoe_xml(xml, dataset=dataset, retrieved_at=RETRIEVED_AT)


@pytest.mark.parametrize("dataset", DATASETS)
def test_verified_request_parameters_and_no_credentials(dataset):
    endpoint, params = build_request(dataset, NORD_EIC, "2025-03-29T23:00Z", "2025-03-30T22:00Z")
    assert endpoint == ENTSOE_ENDPOINT
    assert params["periodStart"] == "202503292300"
    assert params["periodEnd"] == "202503302200"
    assert params["documentType"] == DATASETS[dataset]["document_type"]
    assert "securityToken" not in params
    for domain in DATASETS[dataset]["domain_parameters"]:
        assert params[domain] == NORD_EIC
    assert dataset_metadata(dataset)["historical_feature_eligible"] is False


@pytest.mark.parametrize("dataset", ["IDA1", "imbalance", "continuous_prices"])
def test_unverified_datasets_cannot_silently_substitute_auctions(dataset):
    with pytest.raises(EntsoeAPIError, match="Unsupported"):
        build_request(dataset, NORD_EIC, "2025-01-01T00:00Z", "2025-01-02T00:00Z")


@pytest.mark.parametrize("start,end", [("2025-01-01", "2025-01-02T00:00Z"),
                                       ("2025-01-01T00:00Z", "2025-01-01T00:00Z"),
                                       ("2025-01-01T00:00:01Z", "2025-01-02T00:00Z"),
                                       ("2025-01-01T00:00Z", "2027-01-01T00:00Z")])
def test_request_rejects_ambiguous_or_invalid_boundaries(start, end):
    with pytest.raises(EntsoeAPIError):
        build_request("day_ahead_prices", NORD_EIC, start, end)


def test_native_a03_blocks_decode_repeated_qh_and_preserve_negative_prices():
    result = parse(document_xml())
    assert result.price_eur_mwh.tolist() == [-7, -7, 100, 100]
    assert result.source_position.tolist() == [1, 1, 3, 3]
    assert result.source_block_repeated.tolist() == [False, True, False, True]
    assert (result.product_end - result.product_start).eq(pd.Timedelta(minutes=15)).all()
    assert result.revision.tolist() == ["3"] * 4
    assert result.published_at.isna().all()
    assert result.historical_available_at.isna().all()
    assert not result.historical_feature_eligible.any()
    assert result.source_document_time.eq(pd.Timestamp("2025-01-02T10:00Z")).all()
    assert result.retrieved_at.eq(pd.Timestamp(RETRIEVED_AT)).all()


def test_hourly_prices_stay_hourly_and_a01_gaps_stay_missing():
    periods = period_xml("2025-01-01T00:00Z", "2025-01-01T03:00Z", [(3, 0), (1, -22)], "PT60M")
    result = parse(document_xml(curve="A01", periods=periods))
    assert len(result) == 2
    assert result.price_eur_mwh.tolist() == [-22, 0]
    assert (result.product_end - result.product_start).eq(pd.Timedelta(hours=1)).all()
    assert result.attrs["quality"]["missing_a01_positions"] == 1


@pytest.mark.parametrize("day,expected", [("2025-03-30", 92), ("2025-10-26", 100)])
def test_dst_delivery_days_use_true_utc_length(day, expected):
    first = pd.Timestamp(day, tz="Europe/Rome")
    last = first + pd.DateOffset(days=1)
    periods = period_xml(first.isoformat(), last.isoformat(), [(1, -1)])
    result = parse(document_xml(periods=periods))
    assert len(result) == expected
    assert result.product_start.iloc[0] == first.tz_convert("UTC")
    assert result.product_end.iloc[-1] == last.tz_convert("UTC")
    assert result.product_start.is_unique


def test_disjoint_a03_periods_do_not_fill_gaps_and_mixed_native_mtu_is_retained():
    periods = period_xml("2025-01-01T00:00Z", "2025-01-01T01:00Z", [(1, 4)], "PT60M")
    periods += period_xml("2025-01-01T02:00Z", "2025-01-01T02:30Z", [(1, 6)])
    result = parse(document_xml(periods=periods))
    assert len(result) == 3
    assert result.product_start.tolist() == list(pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T02:00Z", "2025-01-01T02:15Z"]))
    assert result.attrs["quality"]["native_resolutions"] == ["PT15M", "PT60M"]


def test_depublished_value_does_not_inherit_preceding_a03_price():
    periods = period_xml("2025-01-01T00:00Z", "2025-01-01T01:00Z", [(1, -1), (2, None), (4, 5)])
    result = parse(document_xml(periods=periods))
    assert result.price_eur_mwh.iloc[0] == -1
    assert result.price_eur_mwh.iloc[1:3].isna().all()
    assert result.price_eur_mwh.iloc[3] == 5
    assert result.attrs["quality"]["missing_or_depublished_records"] == 2


@pytest.mark.parametrize("dataset", list(DATASETS)[1:])
def test_load_and_generation_units_types_and_unverified_vintage(dataset):
    result = parse(document_xml(dataset=dataset), dataset)
    assert result.value_mw.tolist() == [-7, -7, 100, 100]
    assert result.unit.eq("MW").all()
    assert result.process_type.eq("A01").all()
    assert result.historical_available_at.isna().all()
    assert not result.historical_feature_eligible.any()
    if "psr_type" in DATASETS[dataset]:
        assert result.psr_type.eq(DATASETS[dataset]["psr_type"]).all()


@pytest.mark.parametrize("replacement,message", [
    (("EUR", "USD"), "units"), ((NORD_EIC, "10YIT-GRTN-----B"), "EIC"),
    (("A03</curveType>", "A04</curveType>"), "curves"),
    (("PT15M", "PT5M"), "resolution"),
    (("<revisionNumber>3", "<revisionNumber>0"), "revisionNumber"),
])
def test_parser_rejects_unverified_units_zone_curve_resolution_and_revision(replacement, message):
    with pytest.raises(EntsoeAPIError, match=message):
        parse(document_xml().replace(*replacement))


def test_parser_rejects_realized_load_and_wrong_psr_type():
    with pytest.raises(EntsoeAPIError, match="process"):
        parse(document_xml(dataset="load_forecast").replace("A01</process.processType>", "A16</process.processType>"), "load_forecast")
    with pytest.raises(EntsoeAPIError, match="production type"):
        parse(document_xml(dataset="solar_forecast").replace("B16</psrType>", "B19</psrType>"), "solar_forecast")


def test_intraday_contract_is_never_mislabeled_da():
    xml = document_xml().replace("<curveType>", "<contract_MarketAgreement.type>A07</contract_MarketAgreement.type><curveType>")
    with pytest.raises(EntsoeAPIError, match="non-day-ahead"):
        parse(xml)
    process_xml = document_xml().replace("<createdDateTime>", "<process.processType>A40</process.processType><createdDateTime>")
    with pytest.raises(EntsoeAPIError, match="non-day-ahead"):
        parse(process_xml)


@pytest.mark.parametrize("points", [[(1, 3), (1, 5)], [(0, 1)], [(5, 2)], [(2, 3)]])
def test_positions_are_validated_before_a03_expansion(points):
    periods = period_xml("2025-01-01T00:00Z", "2025-01-01T01:00Z", points)
    with pytest.raises(EntsoeAPIError, match="position"):
        parse(document_xml(periods=periods))


def test_duplicates_across_series_and_periods_are_not_silently_dropped():
    period = period_xml("2025-01-01T00:00Z", "2025-01-01T01:00Z", [(1, 3)])
    with pytest.raises(EntsoeAPIError, match="duplicate"):
        parse(document_xml(periods=period + period))
    second = period_xml("2025-01-01T00:30Z", "2025-01-01T01:30Z", [(1, 3)], "PT60M")
    with pytest.raises(EntsoeAPIError, match="overlapping"):
        parse(document_xml(periods=period + second))


def test_no_data_acknowledgement_and_reflected_secret_are_not_logged():
    no_data = "<Acknowledgement_MarketDocument><Reason><code>999</code><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"
    with pytest.raises(EntsoeNoDataError, match="no matching data"):
        parse(no_data)
    rejected = no_data.replace("No matching data found", "Invalid securityToken=synthetic-test-secret")
    with pytest.raises(EntsoeAPIError) as caught:
        parse(rejected)
    assert "synthetic-test-secret" not in str(caught.value)


@pytest.mark.parametrize("xml", ["<html>Bad gateway</html>", "<broken", '<!DOCTYPE root [<!ENTITY x "value">]><root/>'])
def test_non_xml_or_entity_payload_rejected_without_payload_logging(xml):
    with pytest.raises(EntsoeAPIError):
        parse(xml)


def test_aggregated_generation_consumption_is_explicit_and_not_netted():
    production = document_xml(dataset="generation_forecast")
    series = production.split("<TimeSeries>", 1)[1].split("</TimeSeries>", 1)[0]
    consumption = series.replace("inBiddingZone_Domain.mRID", "outBiddingZone_Domain.mRID").replace("invented-series", "consumption-series")
    result = parse(production.replace("</GL_MarketDocument>", f"<TimeSeries>{consumption}</TimeSeries></GL_MarketDocument>"), "generation_forecast")
    assert len(result) == 8
    assert set(result.direction) == {"production", "consumption"}
