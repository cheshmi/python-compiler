import numpy as np
from numba import njit

from get_ssa_compiler import ssa_by_blocks, blocks, GetSSACompiler, clear, print_ssa
from graph_viz import viz_ast_and_cfg


# dense matrix vector multiplication
@njit(pipeline_class=GetSSACompiler)
def dense_mv(A, x):
    """
    Perform a dense matrix-vector multiplication.

    Args:
        A: A numpy array representing the matrix.
        x: A numpy array representing the vector.

    Returns:
        The result of the matrix-vector multiplication.
    """
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            y[i] = A[i, j] * x[j]
    return y


clear(ssa_by_blocks, blocks)
dense_mv(np.zeros((2, 2)), np.zeros(2))
print_ssa(ssa_by_blocks)
viz_ast_and_cfg(blocks, dense_mv)
