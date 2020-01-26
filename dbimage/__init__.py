import sys
import struct

import blosc
import numpy as np



DTYPE_CODE_LOOKUP = {
    np.dtype('uint8'): 0
}

HDR_FORMAT = "iLLLLLL"


def header_bytes_from_array(array):
    dim = array.shape
    strides = array.__array_interface__['strides']
    assert strides, "Can't handle strides=None"

    dtype = array.dtype
    try:
        dtype_code = DTYPE_CODE_LOOKUP[dtype]
    except KeyError:
        print(f"Can't handle type {dtype}")
        sys.exit(2)

    hdr_values = ((dtype_code,) + dim + strides)
    hdr_packed = struct.pack(HDR_FORMAT, *hdr_values)

    return hdr_packed


def bytes_to_array_info(hdr_bytes):
    hdr_values = struct.unpack(HDR_FORMAT, hdr_bytes)

    dtype_code = hdr_values[0]
    dim = tuple(hdr_values[1:4])
    strides = tuple(hdr_values[4:7])

    return dtype_code, dim, strides


def arraydata_to_compressed_bytes(arraydata):

    cbytes = blosc.compress_ptr(
        arraydata.__array_interface__['data'][0],
        arraydata.size,
        arraydata.dtype.itemsize,
        cname='zlib',
        shuffle=blosc.SHUFFLE
    )

    return cbytes


def compressed_bytes_to_arraydata(cbytes, size, dtype):

    arraydata = np.empty(size, dtype=dtype)
    blosc.decompress_ptr(cbytes, arraydata.__array_interface__['data'][0])

    return arraydata


def compressed_bytes_to_shaped_array(cbytes, dim, strides, dtype=np.uint8):

    rdim, cdim, zdim = dim
    size = rdim * cdim * zdim
    arraydata = compressed_bytes_to_arraydata(cbytes, size, dtype)
    shaped_array = np.lib.stride_tricks.as_strided(arraydata, dim, strides, writeable=False)

    return shaped_array
