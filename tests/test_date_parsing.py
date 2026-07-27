from src.extraction.dates import date_components, parse_dates
def test_numeric_and_missing_year():
    values=parse_dates("Use By 08/14")
    assert values and values[0].value == "08/14"
def test_invalid_date_rejected():
    assert parse_dates("02/31/2027") == []

def test_camera_timestamp_is_rejected():
    assert parse_dates("Jul 25, 2026 at 10:39:55 AM 5180 McGinnis Ferry Rd") == []

def test_month_year_day_dot_matrix_format():
    result = parse_dates("JUL 2026 15")
    assert result[0].value == "JUL 2026 15"
    assert date_components(result[0].value) == (7, 15, 2026)
