import h5py
import numpy as np


def load_h5(path, name, rows=None, cols=None):
    
    with h5py.File(str(path), "r") as f:
        X = f[name]
        if X.ndim == 1:
            out = X[:] if rows is None else X[rows]
        else:
            r = slice(None) if rows is None else rows
            c = slice(None) if cols is None else cols
            out = X[r, c]
        return np.squeeze(np.array(out))


def load_h5_snplist(path, chr_num):

    with h5py.File(path, 'r') as h5f:
        snplist_bytes = h5f['snplist'][:]
        snplist = [s.decode('utf-8') if isinstance(s, bytes) else s for s in snplist_bytes]
        return np.array(snplist)