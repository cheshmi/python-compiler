import ast
from collections import OrderedDict

class SSAInstruction:
    def __init__(self, op, args, result):
        self.op = op  # "add", "phi"
        self.args = args
        self.result = result

    def __repr__(self):
        if self.result:
            return f"{self.result} = {self.op}({', '.join(map(str, self.args))})"
        else:
            return f"{self.op}({', '.join(map(str, self.args))})"
        
    def __eq__(self, other):
        if not isinstance(other, SSAInstruction):
            return False
        return self.op == other.op and self.args == other.args and self.result == other.result


class SSABlock:
    def __init__(self, name):
        self.name = name
        self.instructions = []
        self.successors = []
        self.preds = []

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def __repr__(self):
        block_info = f"\nBlock {self.name}:\n"
        instructions = "\n".join(map(str, self.instructions))
        return block_info + instructions

class SSAConverter(ast.NodeVisitor):
    def __init__(self):
        self.blocks = []
        self.current_block = None 
        self.var_map = {}  
        self.var_counters = {}  
        self.block_counter = 0 
        self.current_def = {} 
        self.incomplete_phis = {}
        self.memoized_expressions = OrderedDict()
        self.cache_size = 100
        self.phi_witnesses = {}

    def new_block(self):
        block = SSABlock(f"block_{self.block_counter}")
        self.block_counter += 1
        self.blocks.append(block)
        return block

    def set_current_block(self, block):
        self.current_block = block

    def add_instruction(self, op, args, result=None):
        instruction = SSAInstruction(op, args, result)
        self.current_block.add_instruction(instruction)
        if op == 'branch':
            # args = [cond, then_name, else_name]
            _, then_name, else_name = args

            then_block = None
            else_block = None
            for block in self.blocks:
                if block.name == then_name:
                    then_block = block
                    break

            for block in self.blocks:
                if block.name == else_name:
                    else_block = block
                    break
            then_block.preds.append(self.current_block)
            else_block.preds.append(self.current_block)
        
        elif op == 'jump':
            # args = [target_name]
            target_name = args[0]

            target_block = None
            for block in self.blocks:
                if block.name == target_name:
                    target_block = block
                    break
            target_block.preds.append(self.current_block)

    def write_variable(self, variable, value):
        if self.current_block.name not in self.current_def:
            self.current_def[self.current_block.name] = {}
        
        self.current_def[self.current_block.name][variable] = value

        self.var_map[variable] = value

        for block in self.blocks:
            for instr in block.instructions:
                if instr.op == "phi" and instr.result == value:
                    definitions = self.current_def[self.current_block.name].values()
                    self.phi_witnesses[value] = tuple(definitions)[:2]
                    break 


    def readVariable(self, var_name, block=None):
        if block:
            block_defs = self.current_def.get(block.name, {})
            if var_name in block_defs:
                return block_defs[var_name]
        
        if block:
            return self.readVariableRecursive(var_name, block)
        
        return None

    def get_new_var(self, var_name):
        if var_name not in self.var_counters:
            self.var_counters[var_name] = 0
        self.var_counters[var_name] += 1
        return f"{var_name}_{self.var_counters[var_name]}"

    def visit_List(self, node):
        elements = [str(self.visit(element)) for element in node.elts]
        return f"[{', '.join(elements)}]"

    def visit_Tuple(self, node):
        elements = [str(self.visit(element)) for element in node.elts]
        return f"({', '.join(elements)})"

    def visit_Set(self, node):
        elements = [str(self.visit(element)) for element in node.elts]
        return f"{{{', '.join(elements)}}}"

    def visit_Dict(self, node):
        keys = [str(self.visit(key)) for key in node.keys]
        values = [str(self.visit(value)) for value in node.values]
        return f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, values))}}}"


    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return f'"{node.value}"'
        elif isinstance(node.value, bool):
            return 'True' if node.value else 'False'
        elif node.value is None:
            return 'None'
        elif isinstance(node.value, complex):
            return str(node.value)
        elif isinstance(node.value, bytes):
            return repr(node.value)
        elif isinstance(node, ast.List):
            return self.visit_List(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, ast.Set):
            return self.visit_Set(node)
        elif isinstance(node, ast.Dict):
            return self.visit_Dict(node)
        else:
            return node.value


    def visit_Name(self, node):
        return self.readVariable(node.id, self.current_block)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        result = None

        for i in range(len(node.ops)):
            op = node.ops[i]
            comparator = node.comparators[i]

            right = self.visit(comparator)
            op_name = type(op).__name__.lower() 
            tmp_result = self.get_new_var("tmp")

            self.add_instruction(op_name, [left, right], tmp_result)

            left = tmp_result

            if result is None:
                result = tmp_result

        return result


    def visit_Assign(self, node):
        value = self.visit(node.value) 
        target = node.targets[0]

        if isinstance(target, ast.Name):
            target_name = target.id
            ssa_var = self.get_new_var(target_name)
            self.add_instruction("assign", [value], ssa_var)
            self.write_variable(target_name, ssa_var)

        elif isinstance(target, ast.Subscript):
            target_obj = self.visit(target.value) 
            target_index = self.visit(target.slice) 
            self.add_instruction("store_element", [target_obj, target_index, value])

        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            elements = target.elts
            if isinstance(value, list):
                if len(elements) != len(value):
                    raise ValueError("Unpacking length mismatch")

                for i, elt in enumerate(elements):
                    if isinstance(elt, ast.Name):
                        ssa_var = self.get_new_var(elt.id)
                        self.add_instruction("assign", [value[i]], ssa_var)
                        self.write_variable(elt.id, ssa_var)

        else:
            raise NotImplementedError(f"Unsupported target type: {type(target).__name__}")


    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            if left == 0:
                return right
            if right == 0:
                return left

        if isinstance(node.op, ast.Mult):
            if left == 0 or right == 0:
                return 0
            if left == 1:
                return right
            if right == 1:
                return left

        if isinstance(node.op, ast.Sub):
            if left == right:
                return 0

        if isinstance(node.op, ast.Div):
            if right == 1:
                return left
            if left == right:
                return 1

        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                result_value = left + right
            elif isinstance(node.op, ast.Sub):
                result_value = left - right
            elif isinstance(node.op, ast.Mult):
                result_value = left * right
            elif isinstance(node.op, ast.Div):
                result_value = left / right
            elif isinstance(node.op, ast.Mod):
                result_value = left % right
            else:
                raise NotImplementedError(f"Unsupported binary operator: {type(node.op).__name__}")

            return result_value

        expr = (left, type(node.op).__name__.lower(), right)

        if expr in self.memoized_expressions:
            self.memoized_expressions.move_to_end(expr)
            return self.memoized_expressions[expr]

        op = type(node.op).__name__.lower()
        result = self.get_new_var("tmp")
        self.add_instruction(op, [left, right], result)
        self.memoized_expressions[expr] = result
        if len(self.memoized_expressions) > self.cache_size:
            self.memoized_expressions.popitem(last=False)

        return result


    def visit_If(self, node):
        cond = self.visit(node.test)
        then_block = self.new_block()
        else_block = self.new_block()
        after_block = self.new_block()

        self.add_instruction("branch", [cond, then_block.name, else_block.name])

        self.set_current_block(then_block)
        self.visit_compound_statement(node.body)
        self.add_instruction("jump", [after_block.name])

        self.set_current_block(else_block)
        self.visit_compound_statement(node.orelse)
        self.add_instruction("jump", [after_block.name])

        self.set_current_block(after_block)

        then_defs = self.current_def.get(then_block.name, {})
        else_defs = self.current_def.get(else_block.name, {})
        all_vars = set(then_defs.keys()).union(else_defs.keys())

        for var in all_vars:
            then_var = then_defs.get(var, None)
            else_var = else_defs.get(var, None)
            
            if then_var is not None and else_var is not None and then_var != else_var:
                phi_var = self.get_new_var(var)
                phi_instr = SSAInstruction("phi", [then_var, else_var], phi_var)
                self.add_instruction(phi_instr.op, phi_instr.args, phi_instr.result)
                self.write_variable(var, phi_var)
                self.removeTrivialPhiRecursively(phi_instr)
        self.set_current_block(after_block)

        self.sealBlock(then_block)
        self.sealBlock(else_block)

    def sealBlock(self, block):
        if block.name in self.incomplete_phis:
            for variable, phi_instr in self.incomplete_phis[block.name].items():
                self.addPhiOperands(variable, phi_instr)
            self.incomplete_phis.pop(block.name, None)


    def visit_While(self, node):
        cond_block = self.new_block()
        body_block = self.new_block()
        after_block = self.new_block()

        self.add_instruction("jump", [cond_block.name])

        self.set_current_block(cond_block)
        cond = self.visit(node.test)
        self.add_instruction("branch", [cond, body_block.name, after_block.name])

        self.set_current_block(body_block)
        initial_var_map = self.var_map.copy()
        self.visit_compound_statement(node.body)
        self.add_instruction("jump", [cond_block.name])

        self.set_current_block(after_block)

        for var in initial_var_map:
            if var in self.var_map and self.var_map[var] != initial_var_map[var]:
                incoming_var = initial_var_map[var]
                body_var = self.var_map[var]
                if incoming_var != body_var:
                    phi_var = self.get_new_var(var)
                    phi_instr = SSAInstruction("phi", [incoming_var, body_var], phi_var)
                    self.add_instruction(phi_instr.op, phi_instr.args, phi_instr.result)
                    self.write_variable(var, phi_var)
                    self.removeTrivialPhiRecursively(phi_instr)
        self.sealBlock(body_block)
        self.sealBlock(cond_block)
        self.sealBlock(after_block)
        

    def visit_For(self, node):
        loop_cond_block = self.new_block()
        loop_body_block = self.new_block()
        after_block = self.new_block()

        iter_obj = self.visit(node.iter)
        
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            # 处理 range(start, stop, step)
            args = [self.visit(arg) for arg in node.iter.args]
            start = args[0] if len(args) > 0 else 0
            stop = args[1] if len(args) > 1 else start
            step = args[2] if len(args) > 2 else 1

            current_var = self.get_new_var("range_index")
            self.add_instruction("assign", [start], current_var)

            iter_var = current_var
            stop_var = stop
            step_var = step
        else:
            iter_var = iter_obj
            index_var = self.get_new_var("index")
            length_var = self.get_new_var("length")
            self.add_instruction("assign", [0], index_var)
            self.add_instruction("length", [iter_var], length_var)
        self.add_instruction("jump", [loop_cond_block.name])
        self.set_current_block(loop_cond_block)
        cond_var = self.get_new_var("loop_cond")


        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range":
            self.add_instruction("lt", [iter_var, stop_var], cond_var)
        else:
            self.add_instruction("lt", [index_var, length_var], cond_var)

        self.add_instruction("branch", [cond_var, loop_body_block.name, after_block.name])

        self.set_current_block(loop_body_block)
        
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range":
            loop_value = iter_var
        else:
            loop_value = self.get_new_var("loop_value")
            self.add_instruction("get_element", [iter_var, index_var], loop_value)

        self.write_variable(node.target.id, loop_value)
        
        self.visit_compound_statement(node.body)

        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range":
            self.add_instruction("add", [iter_var, step_var], iter_var)
        else:
            self.add_instruction("add", [index_var, 1], index_var)

        self.add_instruction("jump", [loop_cond_block.name])

        self.set_current_block(after_block)


    def readVariableRecursive(self, variable, block, visited=None):
        if visited is None:
            visited = set()
        if block.name in visited:
            return None  
        visited.add(block.name)

        if variable in self.current_def.get(block.name, {}):
            return self.current_def[block.name][variable]

        preds = block.preds
        if not preds:
            return None
        elif len(preds) == 1:
            val = self.readVariableRecursive(variable, preds[0], visited)
            self.write_variable(variable, val)
            return val
        else:
            phi_var = self.get_new_var(variable)
            phi_instr = SSAInstruction("phi", [], phi_var)
            self.current_block.add_instruction(phi_instr)
            self.write_variable(variable, phi_var)

            if self.current_block.name not in self.incomplete_phis:
                self.incomplete_phis[self.current_block.name] = {}
            self.incomplete_phis[self.current_block.name][variable] = phi_instr

            operands = []
            for pred in preds:
                operand = self.readVariable(variable, pred)
                operands.append(operand)
            phi_instr.args = operands
        
            if len(operands) >= 2:
                witness = (operands[0], operands[1])
                self.phi_witnesses[phi_var] = witness
            else:
                self.phi_witnesses[phi_var] = tuple(operands)
            
            self.removeTrivialPhiRecursively(phi_instr)
            return phi_var



    def addPhiOperands(self, variable, phi_instr):
        phi_block = self.current_block
        preds = phi_block.preds
        operands = []
        for pred in preds:
            operand = self.readVariable(variable, pred)
            operands.append(operand)
        phi_var = phi_instr.result
        if phi_var in self.phi_witnesses:
            witness = self.phi_witnesses[phi_var]
            if witness[0] != witness[1]:
                phi_instr.args = operands
                return
        else:
            if len(operands) >= 2:
                self.phi_witnesses[phi_var] = (operands[0], operands[1])
        phi_instr.args = operands
        self.removeTrivialPhiRecursively(phi_instr)

    def get_phi_users(self, phi_var):
        users = []
        for block in self.blocks:
            for instr in block.instructions:
                if phi_var in instr.args and instr.op == "phi":
                    users.append(instr)
        return users

    def removeTrivialPhiRecursively(self, phi_instr):
        phi_var = phi_instr.result

        if phi_var in self.phi_witnesses:
            witness = self.phi_witnesses[phi_var]
            if len(witness) >= 2:
                if witness[0] != witness[1]:
                    return
            else:
                if len(set(witness)) == 1:
                    pass
                else:
                    return
        else:
            return
        
        phi_operands = phi_instr.args

        operand_definitions = []
        for op in phi_operands:
            if op != phi_instr.result:
                definition = self.get_definition(op)
                operand_definitions.append(definition)

        if self.are_definitions_identical(operand_definitions):
            same_definition = operand_definitions[0]

            for block in self.blocks:
                for i, instr in enumerate(block.instructions):
                    if instr == phi_instr:
                        new_instruction = SSAInstruction(
                            op='assign',
                            args=same_definition.args,
                            result=phi_instr.result
                        )
                        block.instructions[i] = new_instruction
                        break
            self.var_map[phi_instr.result] = same_definition.result
            #print(self.var_map[phi_instr.result],phi_instr.result)

            users = self.get_phi_users(phi_instr.result)
            for user_instr in users:
                self.removeTrivialPhiRecursively(user_instr)

    def are_definitions_identical(self, definitions):
        if not definitions:
            return False

        reference_definition = definitions[0]

        for definition in definitions[1:]:
            if (reference_definition.op != definition.op or
                reference_definition.args != definition.args):
                return False

        return True

    def get_definition(self, var):
        for block in self.blocks:
            for instr in block.instructions:
                if instr.result == var:
                    return instr
        return None

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

    def visit_Call(self, node):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name == "range":
            args = [self.visit(arg) for arg in node.args]
            return f"range({', '.join(map(str, args))})"
        else:
            raise NotImplementedError(f"Unsupported function call: {func_name}")

    def visit(self, node):
        if isinstance(node, ast.Module):
            block = self.new_block()
            self.set_current_block(block)
            self.visit_compound_statement(node.body)
        elif isinstance(node, ast.Assign):
            self.visit_Assign(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.If):
            self.visit_If(node)
        elif isinstance(node, ast.While):
            self.visit_While(node)
        elif isinstance(node, ast.For):
            self.visit_For(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.Call):  
            return self.visit_Call(node)
        elif isinstance(node, ast.List):
            return self.visit_List(node) 
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)  
        elif isinstance(node, ast.Set):
            return self.visit_Set(node)  
        elif isinstance(node, ast.Dict):
            return self.visit_Dict(node) 
        elif isinstance(node, ast.Expr):
            pass
        else:
            raise NotImplementedError(f"Unsupported AST node type: {type(node).__name__}")

if __name__ == "__main__":
    source_code = """
x = "1"
t = [3,4]
for i in t:
    x = x + i

y = x - x
z = x + y
if z > 2 > x:
    x = y
else:
    x = y
y = 9
while x < 10:
    x = x + 1
use(x)
"""

    source_code1 = """
x = 7
y = 18
x = x + y
"""

    source_code2 = """
x = 1
if x > 0:
    x = 7
else:
    x = 18
"""

    source_code3 = """
x = 1
y = 2
if x > 3:
    x = y
else:
    x = y
"""

    source_code4 = """
y = 1
x = 0
while x < 10:
    x = x + 1
    y = x + 3
"""

    source_code5 = """
x = 0
for i in range(0, 10):
    x = x + i
"""
    source_code6 = """
x = 0
t = [3,4]
for i in t:
    x = x + i
"""
    source_code7 = """
y = 1
x = y - y
"""
    source_code8 = """
y = 2 * 3 + 4 * 6
x = 1
"""
    source_code9 = """
y = 2 
x = y
z = x
"""


    tree = ast.parse(source_code9)
    converter = SSAConverter()
    converter.visit(tree)

    for block in converter.blocks:
        print(block)