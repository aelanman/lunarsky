from astropy.utils.data import (
    download_files_in_parallel,
    export_download_cache,
    import_download_cache,
)
import os
import sys

home = os.path.expanduser("~")
path = os.path.join(home, "ap_cache")
cache_file = os.path.join(path, "astropy_cache.zip")
os.makedirs(path, exist_ok=True)

if sys.argv[1] == "save":
    if not os.path.exists(cache_file):
        knames = [
            "lsk/naif0012.tls",
            "spk/planets/de430.bsp",
            "pck/earth_latest_high_prec.bpc",
        ]
        _naif_kernel_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        kurls = [_naif_kernel_url + kname for kname in knames]
        paths = download_files_in_parallel(kurls, cache=True, show_progress=False)
        export_download_cache(cache_file, urls=kurls, overwrite=True)

if sys.argv[1] == "load":
    import_download_cache(cache_file)
