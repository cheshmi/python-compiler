from numba import njit

from get_ssa_compiler import GetSSACompiler, ssa_by_blocks, blocks, print_ssa, clear
from graph_viz import viz_ast_and_cfg

import numba

from recontruct_ast import construct_ast

print(numba.__version__)
print(numba.config.__dict__)

@njit(pipeline_class=GetSSACompiler)
def test_for_loop():
    x = 0
    for i in range(10):
        for j in range(10):
            x += i * j
    x += 1
    return x



clear(ssa_by_blocks, blocks)
test_for_loop()
print_ssa(ssa_by_blocks)
viz_ast_and_cfg(blocks, test_for_loop)
