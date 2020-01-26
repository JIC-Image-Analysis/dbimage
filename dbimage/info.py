import click

from imageio import imsave

from dbimage.io import read_dbim_from_fpath


@click.command()
@click.argument('dbim_fpath')
def dbim_info(dbim_fpath):

    sa = read_dbim_from_fpath(dbim_fpath)
    print(f"Read array with shape {sa.shape}, dtype {sa.dtype}")

    imsave('z40.png', sa[:,:,40])
