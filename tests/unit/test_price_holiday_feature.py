"""The holiday flag uses delivery-day Italian national dates."""

from datetime import date

from bess_pv_trading.forecasting.italian_holidays import italian_holiday_flags, italian_national_holidays


def test_national_and_moving_holidays():
    holidays = italian_national_holidays(2026)
    assert {date(2026, 4, 6), date(2026, 4, 25), date(2026, 5, 1)} <= holidays
    assert date(2026, 6, 29) not in holidays
    assert date(2025, 4, 21) in italian_national_holidays(2025)


def test_local_delivery_date_controls_flag():
    flags = italian_holiday_flags(['2026-04-24T21:45:00Z', '2026-04-24T22:00:00Z',
                           '2026-05-01T21:45:00Z', '2026-05-01T22:00:00Z'], 'Europe/Rome')
    assert flags.tolist() == [0, 1, 1, 0]
