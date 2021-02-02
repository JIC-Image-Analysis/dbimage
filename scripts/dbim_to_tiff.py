import click
import imageio

import numpy as np

from dbimage.io import read_dbim_from_fpath


@click.command()
@click.argument('dbim_fpath')
@click.argument('tiff_fpath')
def main(dbim_fpath, tiff_fpath):

    im = read_dbim_from_fpath(dbim_fpath)
    im = np.transpose(im, (2, 0, 1))

    imageio.volwrite(tiff_fpath, im)


if __name__ == "__main__":
    main()