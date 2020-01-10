
import numpy as np
import lunarsky
import lunarsky.kernel_manager as kernel_manager


import spiceypy as spice

def test_topo_frame_setup():
    # Check that the frame setup puts all the correct values in the kernel pool.

    latitude, longitude = 30, 25
    name, idnum, frame_dict = kernel_manager.topo_frame_def(latitude, longitude)
    frame_strs = ["{}={}".format(k,v) for (k,v) in frame_dict.items()]
    spice.lmpool(frame_strs)

    for k,v in frame_dict.items():
        N, typecode = spice.dtpool(k)
        if typecode == 'N':
            res = spice.gdpool(k, 0, 100)
            if len(res) == 1:
                res = [int(res[0])]
            if N > 1:
                v = [float(it) for it in v.split(' ')]
                res = res.tolist()
            else:
                res = res[0]
                v = int(v)
            assert v == res
        else:
            res = spice.gcpool(k,0,100)[0]
            v = v.replace('\'', '')
            assert v == res

def test_kernel_paths():
    ## Check that the correct kernel files are downloaded
    ## Need to unhash the file names

    assert len(lunarsky.kernel_manager.KERNEL_PATHS) == 3


