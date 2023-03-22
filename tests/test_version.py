import idesolver


def test_version() -> None:
    assert isinstance(idesolver.__version__, str)
