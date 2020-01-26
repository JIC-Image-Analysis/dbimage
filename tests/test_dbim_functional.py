from tempfile import NamedTemporaryFile

import numpy as np


def test_dbim_functional():

    dim = (256, 256, 30)
    array = np.random.randint(0, 256, dim, dtype=np.uint8)

    from dbimage.io import write_array_to_fpath

    with NamedTemporaryFile() as tmpfile:
        write_array_to_fpath(tmpfile.name, array)

        from dbimage.io import read_dbim_from_fpath
        in_array = read_dbim_from_fpath(tmpfile.name)

    assert (array == in_array).all()
