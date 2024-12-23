import ast
import astor
from graphviz import Digraph

class SSATransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.var_versions = {}

    def visit_Assign(self, node):
        self.generic_visit(node.value)
        
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            if var_name not in self.var_versions:
                self.var_versions[var_name] = 0
            else:
                self.var_versions[var_name] += 1
            target.id = f"{var_name}_{self.var_versions[var_name]}"
        
        return node

    def visit_Name(self, node):
        var_name = node.id
        if var_name in self.var_versions:
            node.id = f"{var_name}_{self.var_versions[var_name]}"
        return node
    
def add_nodes(node, parent=None):
    node_name = f"{type(node).__name__}" 
    dot.node(str(id(node)), node_name)

    if parent:
        dot.edge(str(id(parent)), str(id(node)))

    for child in ast.iter_child_nodes(node):
        add_nodes(child, node)

code = "x = 7 + 18"
tree = ast.parse(code)
print(ast.dump(tree, indent=2))

dot = Digraph()
dot.attr(rankdir="TB") 
add_nodes(tree)
dot.render("ast_graph_tree", format="png", view=False)

code2 = """
x = 7 + 18
x = x * 3
y = "good"
"""
tree2 = ast.parse(code2)
ssa_transformer = SSATransformer()
ssa_tree = ssa_transformer.visit(tree2)
print(astor.to_source(ssa_tree))

code3 = """
a = 0
if True:
    a = 7
else:
    a = 18
b = a + 5
"""

tree3 = ast.parse(code3)
# The SSATransformer generates an incorrect SSA form because the phi function is not implemented.
ssa_transformer = SSATransformer()
ssa_tree = ssa_transformer.visit(tree3)
print(astor.to_source(ssa_tree))

