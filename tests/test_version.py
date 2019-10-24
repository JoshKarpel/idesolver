import pytest

from idesolver.version import _version_info

@pytest.mark.parametrize(
    'version, expected',
    [
        ('0.1.0', (0, 1, 0, None, None)),
        ('0.1.0a5', (0, 1, 0, 'a', 5)),
        ('2.4.3', (2, 4, 3, None, None)),
        ('2.4.3b3', (2, 4, 3, 'b', 3)),
        ('12.44.33', (12, 44, 33, None, None)),
        ('12.44.33b99', (12, 44, 33, 'b', 99)),
    ]
)
def test_version_info(version, expected):
    assert _version_info(version) == expected
