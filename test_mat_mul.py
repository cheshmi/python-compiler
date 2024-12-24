import numpy as np
from numba import njit

from get_ssa_compiler import ssa_by_blocks, blocks, GetSSACompiler, print_ssa, clear
from graph_viz import viz_ast_and_cfg


@njit(pipeline_class=GetSSACompiler)
def matmul(A, B):
    """
    Perform a matrix multiplication.

    Args:
        A: A numpy array representing the first matrix.
        B: A numpy array representing the second matrix.

    Returns:
        The result of the matrix multiplication.
    """
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C



clear(ssa_by_blocks, blocks)
matmul(np.zeros((2, 2)), np.zeros((2, 2)))
print_ssa(ssa_by_blocks)
viz_ast_and_cfg(blocks, matmul)
