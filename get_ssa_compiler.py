from collections import defaultdict

from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.core.untyped_passes import IRProcessing, ReconstructSSA

ssa_by_blocks = defaultdict(list)
blocks = {}
func_ir = None


@register_pass(mutates_CFG=False, analysis_only=True)
class GetSSAPass(FunctionPass):
    _name = "get_ssa"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        blocks.update(state.func_ir.blocks)
        global func_ir
        func_ir = state.func_ir
        for blk_offset, blk in state.func_ir.blocks.items():
            for stmt in blk.body:
                # Group SSA statements by block
                ssa_by_blocks[blk_offset].append(stmt)
        return False  # analysis only


class GetSSACompiler(CompilerBase):

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # pm.add_pass_after(GetSSAPass, ReconstructSSA)
        pm.add_pass_after(GetSSAPass, IRProcessing)
        pm.finalize()
        return [pm]

def clear(ssa_by_blks, blks):
    ssa_by_blks.clear()
    blks.clear()

def print_ssa(ssa_by_blks):
    print("SSA Statements Grouped by Block:")
    for blk_offset, ssa_list in ssa_by_blks.items():
        print(f"Block {blk_offset}:")
        for stmt in ssa_list:
            print(f"  {stmt}")
    print()
