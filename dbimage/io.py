import struct

import numpy as np

from dbimage import (
    HDR_FORMAT,
    bytes_to_array_info,
    compressed_bytes_to_shaped_array
)


def read_dbim_from_fpath(fpath):

    with open(fpath, 'rb') as fh:
        hdr_bytes = fh.read(struct.calcsize(HDR_FORMAT))
        cbytes = fh.read()

    dtype_code, dim, strides = bytes_to_array_info(hdr_bytes)

    shaped_array = compressed_bytes_to_shaped_array(cbytes, dim, strides, np.uint8)

    return shaped_array
