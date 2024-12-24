from numba import njit

from get_ssa_compiler import GetSSACompiler, ssa_by_blocks, blocks, print_ssa, clear
from graph_viz import viz_ast_and_cfg

import numba

print(numba.__version__)
print(numba.config.__dict__)

@njit(pipeline_class=GetSSACompiler)
def test_while_loop():
    x = 0
    while x < 5:
        y = 1
        x += y
        while x % 2 == 0:
            x += 1
        x += 1
    return x



clear(ssa_by_blocks, blocks)
test_while_loop()
print_ssa(ssa_by_blocks)
viz_ast_and_cfg(blocks, test_while_loop)
