# Changelog

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
