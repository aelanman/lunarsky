# lunarsky



[![Build Status](https://travis-ci.org/aelanman/lunarsky.svg?branch=master)](https://travis-ci.org/aelanman/lunarsky)
[![codecov](https://codecov.io/gh/aelanman/lunarsky/branch/master/graph/badge.svg)](https://codecov.io/gh/aelanman/lunarsky)


An extension to `astropy`, providing selenocentric and topocentric reference frames
for the Moon and transformations of star positions from the ICRS system to these
frames. This is to describe the sky as observed from the surface of the Moon.

Non-relativistic transformations are calculated using the SPICE toolkit. Relativistic
corrections (stellar aberration) will be added.


## Dependencies
* `numpy`
* `astropy>=3.0`
* `jplephem`
* `spiceypy`

## Installation

`lunarsky` may be installed with pip:

```
pip install git+https://github.com/aelanman/lunarsky
```

## Usage

![mcmf_coords](./docs/figure.png)

Definition of the MCMF and lunar topocentric frames, from Fig. 2 of [Ye et al.][1]

`lunarsky` provides the following classes:

* `MCMF` – The "Moon-Centered-Moon-Fixed" frame, this is a cartesian reference frame that rotates
with the moon. This is chosen to be the Mean Earth/Polar frame, with a Z axis defined by the mean rotation axis of the Moon and a prime meridian defined by the mean direction to the Earth's center (the X axis is through this meridian 90° from the Z axis, and Y is defined such that XYZ is a right-handed system). This is analogous to `astropy.coordinates.builtin.ITRS`.
* LunarTopo – A topocentric (East/North/Up) frame defined at a position on the Moon's surface. This is analogous to `astropy.coordinates.builtin.AltAz`.
* `MoonLocation` – Analogous to the `astropy.coordinates.EarthLocation` class, this describes
positions on the Moon in either selenocentric (x, y, z) or selenodetic (lat, lon, height) coordinates.
The cartesian axes of the selenocentric system are those of the MCMF frame. In the selenodetic coordinates, "height" is defined relative to a sphere of radius 1737.1 km.
* `SkyCoord` – A replacement for `astropy.coordinates.SkyCoord`, with modifications that ensure compatibility with the `MoonLocation` class.
* `Time` – A replacement for the `astropy.time.Time` class, which can accept a `MoonLocation` instance as its location parameter, allowing for the calculation of local sidereal times on the Moon. The LST of an observer on the Moon is defined as the right ascension of the zenith.


## References
[1]: Ye, Hanlin, et al. "Looking Vector Direction Analysis for the Moon-Based Earth Observation Optical Sensor." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 11, no. 11, Nov. 2018, pp. 4488–99. IEEE Xplore, doi:10.1109/JSTARS.2018.2870247.
