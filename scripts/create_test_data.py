import numpy as np

from tifffile import imwrite


dim = (256, 256, 30)

array = np.random.randint(0, 256, dim, dtype=np.uint8)
tr_array = np.transpose(array, (2, 0, 1))

imwrite('random_sample.tif', tr_array, compress=6)