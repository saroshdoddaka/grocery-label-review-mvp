def test_metrics_importable():
    from src.metrics import summary
    assert set(summary()) == {"total", "reviewed", "unresolved"}
