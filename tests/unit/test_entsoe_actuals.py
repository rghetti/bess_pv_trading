"""Verify realized ENTSO-E series preserve type, native resolution and direction."""

import pandas as pd
import pytest

from bess_pv_trading.data.entsoe import EntsoeAPIError, build_request, parse_entsoe_xml


EIC = '10Y1001A1001A73I'


def series(name, domain, psr='B10'):
    """Build one hourly provider series, including the requested production type."""
    return f'''<TimeSeries><mRID>{name}</mRID><{domain}>{EIC}</{domain}>
    <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name><curveType>A01</curveType>
    <MktPSRType><psrType>{psr}</psrType></MktPSRType><Period><timeInterval>
    <start>2025-10-01T00:00Z</start><end>2025-10-01T02:00Z</end></timeInterval>
    <resolution>PT60M</resolution><Point><position>1</position><quantity>3</quantity></Point>
    <Point><position>2</position><quantity>4</quantity></Point></Period></TimeSeries>'''


def document(content, kind='A75'):
    """Wrap provider series with an actual-data document header."""
    return f'''<GL_MarketDocument><mRID>doc</mRID><revisionNumber>1</revisionNumber>
    <type>{kind}</type><process.processType>A16</process.processType>
    <createdDateTime>2025-10-03T00:00Z</createdDateTime>{content}</GL_MarketDocument>'''


def test_hydro_request_and_distinct_pumping_directions():
    """Hydro generation and pumping must not collapse into one product."""
    _, params = build_request('actual_hydro_pumped_generation', EIC,
                              '2025-10-01T00:00Z', '2025-10-02T00:00Z')
    assert (params['documentType'], params['processType'], params['psrType']) == ('A75', 'A16', 'B10')
    xml = document(series('production', 'inBiddingZone_Domain.mRID') +
                   series('consumption', 'outBiddingZone_Domain.mRID'))
    frame = parse_entsoe_xml(xml, dataset='actual_hydro_pumped_generation', retrieved_at='2025-10-04T00:00Z')
    assert len(frame) == 4
    assert set(frame.direction) == {'production', 'consumption'}
    assert set(frame.psr_type) == {'B10'}
    assert all(frame.product_end - frame.product_start == pd.Timedelta(hours=1))
    assert frame.historical_available_at.isna().all()
    with pytest.raises(EntsoeAPIError, match='production type'):
        parse_entsoe_xml(xml, dataset='actual_hydro_reservoir_generation', retrieved_at='2025-10-04T00:00Z')


def test_realized_load_request_and_parser():
    """Actual load uses A16 and retains load direction."""
    _, params = build_request('actual_load', EIC, '2025-10-01T00:00Z', '2025-10-02T00:00Z')
    assert (params['documentType'], params['processType']) == ('A65', 'A16')
    frame = parse_entsoe_xml(document(series('load', 'outBiddingZone_Domain.mRID'), 'A65'),
                             dataset='actual_load', retrieved_at='2025-10-04T00:00Z')
    assert list(frame.direction) == ['load', 'load']


def test_unfiltered_actual_generation_keeps_distinct_types_and_directions():
    """An unfiltered response must keep hydro production, pumping and gas separate."""
    _, params = build_request('actual_generation_by_type', EIC,
                              '2025-10-01T00:00Z', '2025-10-02T00:00Z')
    assert params['documentType'] == 'A75' and 'psrType' not in params
    xml = document(series('hydro', 'inBiddingZone_Domain.mRID') +
                   series('pumping', 'outBiddingZone_Domain.mRID') +
                   series('gas', 'inBiddingZone_Domain.mRID', 'B04'))
    frame = parse_entsoe_xml(xml, dataset='actual_generation_by_type', retrieved_at='2025-10-04T00:00Z')
    assert len(frame) == 6
    assert set(zip(frame.psr_type, frame.direction)) == {('B10', 'production'), ('B10', 'consumption'), ('B04', 'production')}
