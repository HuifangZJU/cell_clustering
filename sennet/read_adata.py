import pandas as pd
import numpy as np
import h5py

with h5py.File('/media/huifang/data/sennet/xenium/1720/cell_feature_matrix.h5', 'r') as f:
    print(list(f.keys()))