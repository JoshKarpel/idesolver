import re

__version__ = '1.0.4'

VERSION_RE = re.compile(
    r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$',
    re.VERBOSE | re.ASCII,
)


def _version_info(v):
    """Un-format ``__version__``."""
    match = VERSION_RE.match(v)
    if match is None:
        raise Exception(f"Could not determine version info from {v}")

    (major, minor, micro, prerelease, prerelease_num) = match.group(1, 2, 4, 5, 6)

    out = (
        int(major),
        int(minor),
        int(micro or 0),
        prerelease[0] if prerelease is not None else None,
        int(prerelease_num) if prerelease_num is not None else None,
    )

    return out


def version_info():
    """Return a tuple of version information: ``(major, minor, micro, release_level)``."""
    return _version_info(__version__)
