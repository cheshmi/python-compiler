import ast
import operator
import re
from collections import defaultdict

from numba.core import ir
from numba.core.analysis import compute_cfg_from_blocks


def is_loop_entry(label, loops):
    """
    Check if a block(represented by its label) is the loop entry.

    Args:
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        True if the block is the loop entry, false otherwise.
    """
    for _, loop in loops.items():
        if label in loop.entries:
            return True
    return False


def is_in_any_loop(label, loops):
    """
    Check if a block(represented by its label) is part of any loop.

    Args:
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        True if the block is part of any loop, false otherwise.
    """
    for _, loop in loops.items():
        if label in loop.entries or label in loop.body:
            return True
    return False

def compute_unnecessary_variables(block, label, loops, cfg):
    """
    Fold unnecessary statements in the given blocks.

    Args:
        block: Basic block.
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        Set of unnecessary variables
    """
    unnecessary_vars = set()

    if is_loop_entry(label, loops):
        return unnecessary_vars

    if not is_in_any_loop(label, loops):
        return unnecessary_vars

    if isinstance(block.body[-1], ir.Jump):
        return unnecessary_vars

    unnecessary_vars.add(block.body[-1].cond)
    for i in range(-2, -len(block.body) - 1, -1):
        stmt = block.body[i]
        if stmt.target in unnecessary_vars:
            var_list = stmt.list_vars()
            for var in var_list:
                unnecessary_vars.add(var)
    return unnecessary_vars

def convert_to_original_var(ssa_var_name, original_var_list):
    """
    Convert an SSA var back to original version, e.g., x.1 -> x

    Args:
        ssa_var_name: SSA variable name.
        original_var_list: List of original variables.

    Returns:
        Variable name.
    """
    if '$' not in ssa_var_name and '.' in ssa_var_name:
        base_var = ssa_var_name.split('.')[0]
        for original_var in original_var_list:
            if original_var == base_var:
                return original_var
    return ssa_var_name

def convert_stmt_to_node(stmt,
                         inplace_ops,
                         arithmetic_ops,
                         compare_ops,
                         original_vars,
                         ssa_var_map,
                         args):
    """
    Args:
        stmt: SSA statement.
        inplace_ops: Inplace operation map.
        arithmetic_ops: Arithmetic operation map.
        compare_ops: Compare operation map.
        original_vars: Original variable list.
        ssa_var_map: Mapping intermediate SSA variables ($...) to AST node.
        args: Function argument list.

    Returns:
        AST node representing the statement
    """

    def fetch_node_def(var_name, var_map, ast_ctx):
        """
        Fetch node representing var_name. If not found, return var_name as a new node.

        Args:
            var_name: Name of the variable.
            var_map: Mapping var_name to AST node.
            ast_ctx: Context of the AST node.

        Returns:
            AST node.
        """
        if var_name in var_map:
            return var_map[var_name]
        else:
            return ast.Name(id=convert_to_original_var(var_name, original_vars), ctx=ast_ctx)

    # Assign
    if isinstance(stmt, ir.Assign):

        target_name = stmt.target.name

        # These are useless variables.
        if 'phi' in stmt.target.name or 'iter' in stmt.target.name:
            return None

        # Store original variables.
        if '$' not in stmt.target.name and '.' not in stmt.target.name:
            original_vars.append(stmt.target.name)

        # Function argument
        if isinstance(stmt.value, ir.Arg):
            args.append(ast.arg(
                arg=stmt.value.name
            ))

        # R.H.S is Global
        if isinstance(stmt.value, ir.Global):
            if stmt.value.name == 'range':
                ssa_var_map[target_name] = ast.Name(id='range', ctx=ast.Load())
            elif stmt.value.name == 'np':
                ssa_var_map[target_name] = ast.Name(id='np', ctx=ast.Load())
            elif stmt.value.name == 'bool':
                ssa_var_map[target_name] = ast.Name(id='bool', ctx=ast.Load())

        # R.H.S is Expression
        if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'call':
            call_body = stmt.value._kws
            func_name = call_body['func'].name
            if func_name in ssa_var_map:
                func_node = ssa_var_map[func_name]
                call_node = ast.Call(
                    func=func_node,
                    args=[],
                    keywords=[]
                )
                for arg in call_body['args']:
                    call_node.args.append(fetch_node_def(arg.name, ssa_var_map, ast.Load()))
                if isinstance(func_node, ast.Name):
                    if func_node.id == 'bool':
                        return return_or_store(target_name,
                                               fetch_node_def(call_body['args'][0].name, ssa_var_map, ast.Load()),
                                               ssa_var_map)
                    elif func_node.id == 'range':
                        return ast.For(
                            target=None,
                            iter=call_node,
                            body=[],
                            orelse=[]
                        )
                elif isinstance(func_node, ast.Attribute):
                    return ast.Assign(
                        targets=[ast.Name(id=target_name, ctx=ast.Store())],
                        value=call_node
                    )

        # R.H.S of Assign is a Variable
        if isinstance(stmt.value, ir.Var):
            if 'phi' in stmt.value.name or 'iter' in stmt.value.name:
                return None
            value_node = fetch_node_def(stmt.value.name, ssa_var_map, ast.Load())
            if isinstance(value_node, ast.AugAssign):
                return value_node
            else:
                return ast.Assign(
                    targets=[fetch_node_def(target_name, ssa_var_map, ast.Load())],
                    value=value_node
                )

        # R.H.S of Assign is Const
        elif isinstance(stmt.value, ir.Const):
            const_node = ast.Constant(value=stmt.value.value)
            return return_or_store(target_name, const_node, ssa_var_map)

        # R.H.S of Assign is a binop
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop':
            # binop is an arithmetic operation
            op_node = None
            if stmt.value.fn in arithmetic_ops:
                op_node = ast.BinOp(
                    left=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Load()),
                    op=arithmetic_ops[stmt.value.fn],
                    right=fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load()),
                )
            # binop is a compare operation
            if stmt.value.fn in compare_ops:
                op_node = ast.Compare(
                    left=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Load()),
                    ops=[compare_ops[stmt.value.fn]],
                    comparators=[fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load())],
                )
            if op_node is not None:
                return return_or_store(target_name, op_node, ssa_var_map)
            else:
                raise ValueError("Unsupported binary operation!")

        # R.H.S of Assign is inplace binop, e.g., x += 1 -> $tmp = inplace(add, x, 1)
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'inplace_binop':
            aug_assign_node = ast.AugAssign(
                target=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Store()),
                op=inplace_ops[stmt.value.fn],
                value=fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load()),
            )
            if '$' not in target_name:
                return aug_assign_node
            else:
                return return_or_store(target_name, aug_assign_node, ssa_var_map)

        # Expr
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'cast':
            var_node = ast.Name(id=convert_to_original_var(stmt.value._kws['value'].name, original_vars), ctx=ast.Load())
            return return_or_store(target_name, var_node, ssa_var_map)

        # Build tuple, e.g., [i, j] of A[i, j]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'build_tuple':
            tuple_node = ast.Tuple(elts=[], ctx=ast.Load())
            for var in stmt.value.items:
                tuple_node.elts.append(fetch_node_def(var.name, ssa_var_map, ast.Load()))
            return return_or_store(target_name, tuple_node, ssa_var_map)

        # Subscripting, e.g., A[i, j]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getitem":
            value_node = fetch_node_def(stmt.value.value.name, ssa_var_map, ast.Load())
            slice_node = fetch_node_def(stmt.value.index.name, ssa_var_map, ast.Load())
            subscript_node = ast.Subscript(
                value=value_node,
                slice=slice_node,
                ctx=ast.Load()
            )
            return return_or_store(target_name, subscript_node, ssa_var_map)

        # Get attribute, e.g., A.shape, np.zeros
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getattr":
            attribute_node = ast.Attribute(
                value=fetch_node_def(stmt.value.value.name, ssa_var_map, ast.Load()),
                attr=stmt.value.attr,
                ctx=ast.Load()
            )
            return return_or_store(target_name, attribute_node, ssa_var_map)

    # Deal with y[i] = ...
    elif isinstance(stmt, ir.SetItem):
        value_node = fetch_node_def(stmt.value.name, ssa_var_map, ast.Load())
        if isinstance(value_node, ast.AugAssign):
            return value_node
        else:
            subscript_node = ast.Subscript(
                value=ast.Name(id=stmt.target.name, ctx=ast.Load()),
                slice=ast.Name(id=stmt.index.name, ctx=ast.Load()),
                ctx=ast.Load()
            )
            return ast.Assign(
                targets=[subscript_node],
                value=value_node
            )

    # Return
    elif isinstance(stmt, ir.Return):
        var_node = fetch_node_def(stmt.value.name, ssa_var_map, ast.Load())
        return ast.Return(value=var_node)

    return None

def return_or_store(var_name, ast_node, var_map):
    """
    Return ast.Assign of ast_node, or store it if the node is part of another node.

    Args:
        var_name: Name of the variable.
        ast_node: AST node.
        var_map: Mapping var_name to AST node.

    Returns:
        AST node.
    """
    if "$" in var_name:
        var_map[var_name] = ast_node
        return None
    else:
        return ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Load())],
            value=ast_node
        )

def insert_while(cond_stmt_node):
    """
    Args:
        Condition statement node

    Returns:
        AST node representing the branch statement
    """
    return ast.While(
        test=cond_stmt_node,  # Right subtree of the condition statement node
        body=[],  # Loop body is the node converted from the body block
        orelse=[]
    )

def insert_if(cond_stmt_node):
    return ast.If(
        test=cond_stmt_node.value,
        body=[],
        orelse=[]
    )

def process_for_node(label, loops, cfg, blocks, for_node):
    """
    Find the loop variant variable of the AST For node.
    e.g. var i of the for loop: for i in range(10)

    Args:
        label: Label of the basic block.
        loops: A dictionary mapping loop headers to loop information.
        cfg: CFG.
        blocks: Basic blocks.
        for_node: AST For node to be processed.

    Returns:
        Processed For node.
    """
    loop_header_label = int()
    for src, _ in cfg.successors(label):
        if src in loops:
            loop_header_label = src
            break
    true_condition_label = blocks[loop_header_label].body[-1].truebr
    loop_var = blocks[true_condition_label].body[0].target

    for_node.target = ast.Name(
        id=loop_var,
        ctx=ast.Load()
    )

    return for_node

def convert_block_to_ast(label, loops, cfg, blocks, args, original_vars):
    """
    Args:
        label: Label of the basic block.
        loops: A dictionary mapping loop headers to loop information.
        cfg: CFG.
        blocks: Basic blocks.
        args: Arguments of the function.

    Returns:
        List of AST nodes representing the block
    """
    inplace_ops = {
        operator.iadd: ast.Add(),
        operator.isub: ast.Sub(),
        operator.imul: ast.Mult(),
        operator.itruediv: ast.Div(),
        operator.imod: ast.Mod()
    }

    arithmetic_ops = {
        operator.add: ast.Add(),
        operator.sub: ast.Sub(),
        operator.mul: ast.Mult(),
        operator.truediv: ast.Div(),
        operator.floordiv: ast.FloorDiv(),
        operator.mod: ast.Mod()
    }

    compare_ops = {
        operator.lt: ast.Lt(),
        operator.le: ast.LtE(),
        operator.eq: ast.Eq(),
        operator.ne: ast.NotEq(),
        operator.ge: ast.GtE(),
        operator.gt: ast.Gt()
    }

    block = blocks[label]

    ast_nodes = list()
    unnecessary_vars = compute_unnecessary_variables(block, label, loops, cfg)
    ssa_var_map = defaultdict()

    # Convert stmt to AST node
    for stmt in block.body:
        if not is_loop_entry(label, loops):
            if isinstance(stmt, ir.Assign) and stmt.target in unnecessary_vars:
                continue
            if isinstance(stmt, ir.Branch) and stmt.cond in unnecessary_vars:
                continue
        ast_node = convert_stmt_to_node(stmt,
                                        inplace_ops,
                                        arithmetic_ops,
                                        compare_ops,
                                        original_vars,
                                        ssa_var_map,
                                        args)
        if ast_node is not None:
            if isinstance(ast_node, ast.For):
                for_node = process_for_node(label, loops, cfg, blocks, ast_node)
                ast_nodes.append(for_node)
            else:
                ast_nodes.append(ast_node)

    # Deal with Branch statement
    # Branch could be in loop entry or body
    # Branch must be in the last line in a block
    if isinstance(block.body[-1], ir.Branch):
        # If the block is loop entry
        if is_loop_entry(label, loops):
            cond_var = block.body[-1].cond.name
            cond_stmt_node = ssa_var_map[cond_var]
            ast_nodes.append(insert_while(cond_stmt_node))

    return ast_nodes


def insert_temporary_node(label, ast_nodes):
    """
    For testing only. Insert a temporary parent node for a list
    of AST nodes.

    Args:
        ast_nodes: List of AST nodes of the block.
        label: Label of the block.

    Returns:
        AST node
    """
    temp_node = ast.FunctionDef(
        name="Block {}".format(label),
        args=None,
        body=ast_nodes,
        decorator_list=[]
    )

    return temp_node


def construct_ast(blocks):
    """
    Args:
        blocks: Basic blocks.

    Returns:
        AST node representing the program
    """
    cfg = compute_cfg_from_blocks(blocks)
    loop_header_to_body = cfg._find_loops()
    arg_list = list()
    original_vars = list()

    # Compute AST node list of each block
    all_ast_nodes = defaultdict(list)
    for label, block in blocks.items():
        ast_nodes = convert_block_to_ast(label, loop_header_to_body, cfg, blocks, arg_list, original_vars)
        all_ast_nodes[label] = ast_nodes

    loop_entry_to_body = defaultdict(list)
    loop_entry_to_header = defaultdict()
    for header, loop in loop_header_to_body.items():
        for entry in loop.entries:
            loop_entry_to_body[entry] = loop.body
            loop_entry_to_header[entry] = header

    # Compute dominator tree
    dom_tree = cfg.dominator_tree()

    def recursive_construct(block_label, depth=0):
        """
        Recursively construct AST.

        Args:
            block_label: Label of the basic block.
            depth: AST depth.

        Returns:
            AST node list.
        """
        nodes = all_ast_nodes[block_label]  # List of AST nodes of current block
        child_labels = dom_tree.get(block_label, set())  # Fetch dominated children blocks

        # If current block is loop entry
        if block_label in loop_entry_to_body:
            loop_node = nodes[-1]   # Last node must be loop node
            if isinstance(loop_node, ast.For):
                for_header_label = list(dom_tree[block_label])[0]
                dom_tree[block_label] = dom_tree[for_header_label]
                del dom_tree[for_header_label]
                child_labels = dom_tree.get(block_label, set())

            # Process blocks in loop body
            for child_label in child_labels:
                # If child block belongs to loop body
                if child_label in loop_entry_to_body[block_label]:
                    loop_node.body.extend(recursive_construct(child_label, depth + 1))
                else:
                    # Skip non-loop-body blocks for now
                    pass

        # Process same-depth dominated children blocks
        for child_label in child_labels:
            if child_label not in loop_entry_to_body.get(block_label, set()):  # not in loop body
                nodes.extend(recursive_construct(child_label, depth))

        return nodes

    # Function body
    func_body = recursive_construct(0)

    # Function arguments
    args = ast.arguments(
        posonlyargs=[],
        args=arg_list,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )

    # Function node
    func_def = ast.FunctionDef(
        name="func",
        args=args,
        body=func_body,
        decorator_list=[],
        returns=None
    )

    # Root node
    module = ast.Module(
        body=[func_def],
        type_ignores=[]
    )

    return module
