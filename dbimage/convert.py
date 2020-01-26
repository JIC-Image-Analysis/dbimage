import logging

import click
import numpy as np
from tifffile import TiffFile

from dbimage import arraydata_to_compressed_bytes, header_bytes_from_array


def arraydata_from_tif_fpath(tif_fpath):

    with TiffFile(tif_fpath) as tif:
        rawdata = tif.asarray()

    arraydata = np.transpose(rawdata, (1, 2, 0))

    return arraydata


@click.command()
@click.argument('input_tif_fpath')
@click.argument('output_dbim_fpath')
def convert_tif_image(input_tif_fpath, output_dbim_fpath):

    arraydata = arraydata_from_tif_fpath(input_tif_fpath)
    print(arraydata.__array_interface__['strides'])
    logging.info(f"Loaded TIFF with shape {arraydata.shape}, dtype {arraydata.dtype}")

    cbytes = arraydata_to_compressed_bytes(arraydata)
    hdr_bytes = header_bytes_from_array(arraydata)

    with open(output_dbim_fpath, 'wb') as fh:
        fh.write(hdr_bytes)
        fh.write(cbytes)
