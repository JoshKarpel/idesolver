# Changelog

## `1.1.0`

Released `2022-11-19`

### Added

- [#35](https://github.com/JoshKarpel/idesolver/pull/35](https://github.com/nbrucy)) Add support for multidimensional IDEs, by [@nbrucy](https://github.com/nbrucy).

## `1.0.5`

Released `2020-09-15`

### Changed

- Relaxed dependency version restrictions in advance of changes to `pip`.
  There shouldn't be any impact on users.

## `1.0.4`

Released `2019-10-24`

### Changed

- Revision of packaging and CI flow. There shouldn't be any impact on users.

## `1.0.3`

Released `2022-02-27`

### Changed

- Revision of package structure and CI flow. There shouldn't be any impact on users.

## `1.0.2`

Released `2018-01-30`

### Changed

- IDESolver now explicitly requires Python 3.6+. Dependencies on `numpy` and `scipy` are given as lower bounds.

## `1.0.1`

Released `2018-01-14`

### Changed

- Changed the name of `IDESolver.F` to `IDESolver.f`, as intended.
- The default global error function is now injected instead of hard-coded.
