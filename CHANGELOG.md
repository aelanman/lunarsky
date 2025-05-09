# Changelog

## [0.2.6] -- 2024-12-12
## Fixed
- Avoid deprecation warning with numpy.broadcast_shape

## [0.2.5] -- 2024-06-26

## Fixed
- Maintain compatibility with astropy 6.0.0
- Compatibility with numpy 2.0


## [0.2.3] -- 2024-05-13

## Fixed
- Make lunarsky.time.Time location handling compatible with astropy >= 6.1

## [0.2.2] -- 2024-03-19

## Added
- Support for ellipsoid models for selenodetic coordinates (non-spherical)

## Deprecated
- Removed support for Python < 3.9. Astropy >= 6.0 is required for ellipsoid support

## [0.2.0] -- 2022-10-12

## Fixed
- Updated version of pre-commit-hooks used
- Accept tuple for location in Time class (in this case, assumes EarthLocation)
- Use newer PCK file in unit test with Earth positioning.
- Match behavior of astropy when transforming non-unit cartesian positions without units (treat as unitspherical using direction info only)
- Now tracking available lunar station_ids, instead of incrementing a counter naively.

## Deprecated
- Dropping support for Python 3.6

## [0.1.2] -- 2022-01-17

## Added
- Support for having multiple LunarTopo frames at once
- Deletion of old LunarTopo frame variables from kernel pool when MoonLocation deleted
- Cleaned up code in topo.py to reduce code duplication.

## Fixed
- Transformations involving MoonLocation objects holding multiple positions.
- Transformations from LunarTopo to LunarTopo (at a different place).

## Deprecated

## [0.1.1] -- 2022-01-15

## Added
- Corrected transformations for nearby objects (in the solar system)

## Fixed
- Bug related to multiple obstimes in topo transformations

## [0.0.2] -- 2020-06-07

## Added
- More references to README

## Changed
- Keep pck and fk kernels in the repository, instead of caching.

## [0.0.1] -- 2020-05-22
- Initial release.
