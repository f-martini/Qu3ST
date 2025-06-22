import numpy as np


def bitfield(n, L):
    if isinstance(n, int):
        result = np.binary_repr(n, L)
    else:
        # it is already a binary string
        result = n
    return [int(digit) for digit in result]


def add_barrier(qc, flag):
    if flag:
        qc.barrier()
