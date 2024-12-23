# README for SSA Conversion and Optimization Framework

## Overview

This project implements a framework for converting Python code into Static Single Assignment (SSA) form by parsing the Abstract Syntax Tree (AST). It includes various optimization techniques to enhance the quality and efficiency of the generated code.

### Features
1. **AST to SSA Conversion**:
   - Efficiently converts Python AST into SSA form.
   - Supports basic assignments, arithmetic operations, conditional branching, and loops.

2. **Optimization Techniques**:
   - Local Value Numbering (LVN)
   - Global Value Numbering (GVN)
   - Arithmetic simplifications
   - Constant folding
   - Removal of trivial Phi functions
   - Common subexpression elimination with caching

3. **Code Efficiency**:
   - Reduces redundant instructions.
   - Dynamically handles variable tracking with Phi function insertion.

---

## Code Structure

- **`SSAInstruction`**:
  Represents individual SSA instructions such as `add`, `phi`, and `assign`.

- **`SSABlock`**:
  Represents a basic block in the SSA form, containing instructions and control flow details.

- **`SSAConverter`**:
  Core class responsible for:
  - Parsing Python AST nodes.
  - Generating SSA instructions.
  - Performing optimizations during SSA construction.

---

## Requirements

- Python 3.7+
- Standard libraries: `ast`, `collections`

---

## Usage

### Running the Framework
The framework includes several predefined source code examples to test SSA conversion. To run the script:

```bash
python project2.py
```

