"""Synthetic documented-format GME fixtures; no live response is implied."""
import base64
from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.data.gme import (
    DATA_URL, GMEAPIError, GMENoDataError, GMESensitiveResponseError,
    build_request, dataset_metadata, parse_gme_response, parse_token_response,
    validate_sanitized_response,
)


def envelope(records, *, filename="prices.json", extra_members=None, outer=None):
    """Package synthetic records exactly as the documented JSON ZIP transport."""
    stream = BytesIO()
    body = records if isinstance(records, str) else json.dumps(records)
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, body)
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    result = {"RequestId": "synthetic-documentation-fixture", "FormatType": ".json.zip",
              "ResultRequest": "", "ContentResponse": base64.b64encode(stream.getvalue()).decode("ascii")}
    result.update(outer or {})
    return json.dumps(result).encode()


def records(day="20251002", market="MGP", *, count=96, offset=0):
    """Use chronological flow-hour/period ordinals, preserving negative prices."""
    return [{"FlowDate": int(day), "Hour": (period - 1) // 4 + 1,
             "Period": str(period), "Market": market, "Zone": "NORD", "Price": period - 50., "Notes": ""}
            for period in range(1 + offset, 1 + offset + count)]


def parse(rows, **kwargs):
    """Normalize one fixture without network requests or historical publication claims."""
    options = {"dataset": "day_ahead_prices", "retrieved_at": "2026-10-16T12:00:00Z"}
    options.update(kwargs)
    return parse_gme_response(envelope(rows), **options)


def test_nonsecret_post_request_uses_local_inclusive_dates_and_documented_fields():
    url, body = build_request("ida1_prices", "2025-03-29T23:00:00Z", "2025-03-30T22:00:00Z")
    assert url == DATA_URL
    assert body == {"Platform": "PublicMarketResults", "Segment": "MI-A1", "DataName": "ME_ZonalPrices",
                    "IntervalStart": 20250330, "IntervalEnd": 20250330, "Attributes": {}}
    assert not any(name.lower() in {"login", "password", "token"} for name in body)
    assert not dataset_metadata("ida3_prices")["historical_feature_eligible"]


def test_native_quarter_prices_preserve_values_and_leave_publication_unknown():
    frame = parse(records())
    assert len(frame) == 96
    assert frame.price_eur_mwh.iloc[0] == -49.
    assert frame.product_start.iloc[0] == pd.Timestamp("2025-10-01T22:00Z")
    assert frame.product_end.iloc[-1] == pd.Timestamp("2025-10-02T22:00Z")
    assert set(frame.native_interval_minutes) == {15}
    assert frame.historical_available_at.isna().all()
    assert frame.published_at.isna().all()
    assert not frame.historical_feature_eligible.any()
    assert frame.record_status.eq("observed").all()
    assert set(frame.auction) == {"DA"}


def test_day_ahead_before_october_stays_hourly_and_is_not_expanded_to_quarters():
    hourly = [{"FlowDate": 20250102, "Hour": hour, "Period": str(hour), "Market": "MGP",
               "Zone": "NORD", "Price": -20 + hour, "Notes": None} for hour in range(1, 25)]
    frame = parse(hourly)
    assert len(frame) == 24
    assert set(frame.native_interval_minutes) == {60}
    assert ((frame.product_end - frame.product_start).dt.total_seconds() == 3600).all()
    missing_period = [{key: value for key, value in row.items() if key != "Period"} for row in hourly]
    assert len(parse(missing_period)) == 24


@pytest.mark.parametrize("day,count,minutes,market,dataset", [
    ("20250330", 23, 60, "MGP", "day_ahead_prices"),
    ("20251026", 100, 15, "MGP", "day_ahead_prices"),
    ("20250330", 92, 15, "MI-A1", "ida1_prices"),
    ("20251026", 100, 15, "MI-A2", "ida2_prices"),
])
def test_dst_native_products_follow_actual_23_and_25_hour_local_days(day, count, minutes, market, dataset):
    rows = records(day, market, count=count)
    if minutes == 60:
        for index, row in enumerate(rows):
            row["Hour"], row["Period"] = index + 1, str(index + 1)
    frame = parse(rows, dataset=dataset)
    assert len(frame) == count
    assert set(frame.native_interval_minutes) == {minutes}
    assert not frame.product_start.duplicated().any()
    assert (frame.product_end.max() - frame.product_start.min()).total_seconds() == count * minutes * 60
    if day == "20251026":
        repeated = frame.loc[frame.product_start.dt.tz_convert("Europe/Rome").dt.hour == 2]
        assert len(repeated) == 8


def test_ida_is_native_quarter_hour_in_january_and_ida3_is_afternoon_only_on_dst_days():
    assert len(parse(records("20250102", "MI-A1"), dataset="ida1_prices")) == 96
    # Spring loses four morning products, so noon starts at global period 45.
    spring = parse(records("20250330", "MI-A3", count=48, offset=44), dataset="ida3_prices")
    assert len(spring) == 48
    assert spring.product_start.iloc[0].tz_convert("Europe/Rome").hour == 12
    # Autumn gains four morning products; noon starts at global period 53.
    autumn = parse(records("20251026", "MI-A3", count=48, offset=52), dataset="ida3_prices")
    assert len(autumn) == 48
    assert autumn.product_start.iloc[0].tz_convert("Europe/Rome").hour == 12
    with pytest.raises(GMEAPIError, match="ineligible morning"):
        parse(records("20250102", "MI-A3", count=1), dataset="ida3_prices")


def test_gaps_and_missing_prices_are_preserved_without_filling_native_labels():
    rows = records(count=3)
    rows[2]["Price"] = None
    frame = parse([rows[0], rows[2]])
    assert len(frame) == 2
    assert frame.source_period.tolist() == [1, 3]
    assert np.isnan(frame.price_eur_mwh.iloc[1])
    assert frame.record_status.iloc[1] == "missing_price"
    assert frame.attrs["missing_price_rows"] == 1


def test_empty_zone_absence_and_cancelled_auction_are_no_data_never_zero_prices():
    with pytest.raises(GMENoDataError, match="empty"):
        parse([])
    with pytest.raises(GMENoDataError, match="requested zone"):
        parse([{**records(count=1)[0], "Zone": "SUD"}])
    with pytest.raises(GMENoDataError, match="cancelled"):
        parse([{**records(count=1)[0], "Price": 0, "Notes": "Auction cancelled"}])
    with pytest.raises(GMENoDataError, match="cancelled"):
        parse([{"FlowDate": 20251002, "Market": "MGP", "Zone": "NORD", "Notes": "Auction cancelled"}])
    with pytest.raises(GMEAPIError, match="mixes cancelled"):
        parse([records(count=1)[0], {"FlowDate": 20251002, "Market": "MGP", "Zone": "NORD", "Notes": "Auction cancelled"}])


@pytest.mark.parametrize("change,match", [
    ({"Period": "97"}, "Period/Hour"), ({"Period": "2", "Hour": 2}, "Period/Hour"),
    ({"Period": "1.5"}, "Period"), ({"Hour": True}, "Hour"), ({"Market": "MI-A1"}, "market"),
    ({"FlowDate": 20250230}, "FlowDate"), ({"Price": "nan"}, "Price"),
    ({"Price": True}, "Price"), ({"Notes": {}}, "Notes"),
])
def test_malformed_native_records_are_rejected_without_echoing_values(change, match):
    with pytest.raises(GMEAPIError, match=match):
        parse([{**records(count=1)[0], **change}])


def test_duplicate_products_requested_bounds_and_unsafe_granularity_are_rejected():
    with pytest.raises(GMEAPIError, match="duplicate native"):
        parse(records(count=1) * 2)
    with pytest.raises(GMEAPIError, match="outside the requested"):
        parse(records(count=1), start="2025-10-02T22:00Z", end="2025-10-03T22:00Z")
    with pytest.raises(GMEAPIError, match="native GME granularity"):
        build_request("day_ahead_prices", "2025-10-01T22:00Z", "2025-10-02T22:00Z", granularity_type="PT60")
    with pytest.raises(GMEAPIError, match="only from 2025-10-01"):
        build_request("day_ahead_prices", "2025-01-01T23:00Z", "2025-01-02T23:00Z", granularity_type="PT60")
    with pytest.raises(GMEAPIError, match="calendar days"):
        build_request("ida1_prices", "2025-10-01T12:00Z", "2025-10-02T12:00Z")
    with pytest.raises(GMEAPIError, match="offset-aware"):
        build_request("ida1_prices", "2025-10-01", "2025-10-02")


def test_authentication_response_token_is_memory_only_and_failure_reasons_are_sanitized():
    assert parse_token_response(b'{"Success":true,"token":"header.payload.signature","Reason":""}') == "header.payload.signature"
    with pytest.raises(GMEAPIError) as caught:
        parse_token_response('{"Success":false,"token":"","Reason":"password=do-not-leak"}')
    assert "do-not-leak" not in str(caught.value)
    with pytest.raises(GMEAPIError, match="valid nonempty token"):
        parse_token_response('{"Success":true,"token":"bad\\nheader"}')


@pytest.mark.parametrize("place", ["envelope", "zip", "escaped", "nested_escaped"])
def test_known_secrets_are_rejected_in_raw_envelope_and_decoded_zip_before_cache(place):
    secret = "private-login-value" if place not in {"escaped", "nested_escaped"} else "private-\u00e8-value"
    notes = json.dumps({"echo": secret}, ensure_ascii=True) if place == "nested_escaped" else secret
    body = (envelope(records(count=1), outer={"ResultRequest": secret}) if place == "envelope" else
            envelope([{**records(count=1)[0], "Notes": notes}]))
    with pytest.raises(GMESensitiveResponseError) as caught:
        validate_sanitized_response(body, sensitive_values=[secret])
    assert secret not in str(caught.value)
    with pytest.raises(GMESensitiveResponseError):
        parse_gme_response(body, dataset="day_ahead_prices", retrieved_at="2026-10-16T00:00Z", sensitive_values=[secret])


def test_unexpected_credential_fields_are_rejected_even_without_known_secret_values():
    for body in (envelope(records(count=1), outer={"token": "unexpected"}),
                 envelope([{**records(count=1)[0], "nested": {"Password": "unexpected"}}])):
        with pytest.raises(GMESensitiveResponseError):
            validate_sanitized_response(body)
    validate_sanitized_response(envelope(records(count=1)))


@pytest.mark.parametrize("location", ["filename", "archive_comment", "member_comment", "member_extra"])
def test_known_secrets_in_archive_metadata_are_never_safe_to_cache(location):
    secret = "synthetic-private-\u00e8-metadata"
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        member = zipfile.ZipInfo(secret + ".json" if location == "filename" else "prices.json")
        if location == "member_comment":
            member.comment = secret.encode()
        elif location == "member_extra":
            value = secret.encode()
            member.extra = b"\xfe\xca" + len(value).to_bytes(2, "little") + value
        elif location == "archive_comment":
            archive.comment = secret.encode()
        archive.writestr(member, json.dumps(records(count=1)))
    body = json.dumps({"FormatType": ".json.zip", "ContentResponse": base64.b64encode(stream.getvalue()).decode()})
    with pytest.raises(GMESensitiveResponseError) as caught:
        validate_sanitized_response(body, sensitive_values=[secret])
    assert secret not in str(caught.value)


@pytest.mark.parametrize("filename", ["../prices.json", "/prices.json", "C:/prices.json", "folder\\prices.json", "prices.xml"])
def test_zip_paths_are_validated_and_nothing_is_extracted(filename):
    with pytest.raises(GMEAPIError, match="member path"):
        parse_gme_response(envelope(records(count=1), filename=filename),
            dataset="day_ahead_prices", retrieved_at="2026-10-16T00:00Z")


def test_zip_and_json_limits_invalid_encoding_and_duplicate_fields_are_rejected():
    body = envelope(records(count=2))
    for limits in ({"max_response_bytes": 10}, {"max_archive_bytes": 10}, {"max_uncompressed_bytes": 10}, {"max_records": 1}):
        with pytest.raises(GMEAPIError):
            validate_sanitized_response(body, **limits)
    with pytest.raises(GMEAPIError, match="exactly one JSON"):
        validate_sanitized_response(envelope(records(count=1), extra_members={"other.json": "[]"}))
    with pytest.raises(GMEAPIError, match="base64"):
        validate_sanitized_response(json.dumps({"FormatType": ".json.zip", "ContentResponse": "???"}))
    with pytest.raises(GMEAPIError, match="ZIP"):
        validate_sanitized_response(json.dumps({"FormatType": ".json.zip", "ContentResponse": base64.b64encode(b"bad zip").decode()}))
    with pytest.raises(GMEAPIError, match="duplicate object"):
        validate_sanitized_response(envelope('[{"FlowDate":20251002,"FlowDate":20251003}]'))
    with pytest.raises(GMEAPIError, match="bounded array"):
        validate_sanitized_response(envelope({"data": records(count=1)}))
