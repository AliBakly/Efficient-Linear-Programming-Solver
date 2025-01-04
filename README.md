# C Implementation of Linear and Integer Programming Solver

This repository contains an implementation of a linear programming solver that utilizes the Simplex algorithm and Branch-and-Bound method for integer solutions.

## Project Structure

```
.
├── intopt.c          # Main implementation file
├── README.md         # This file
└── tests/            # Test cases directory
    ├── 1/           # Test suite 1 
    ├── 2/           # Test suite 2
    └── ...          # Additional test suites
```

Each test directory contains:
- `a.lp`: Linear programming problem specification
- `a.sol`: Expected solution
- `i`: Input file for the solver
- `intopt.sol`: Solution produced by our implementation
- `gurobi.log`, `gurobi.out`: Reference solutions from Gurobi solver

## Implementation Details

The solver implements two main algorithms:
1. **Simplex Algorithm**: Solves linear programming problems through iterative improvement
2. **Branch-and-Bound**: Extends the simplex solver to find integer solutions

Key features include:
- Robust pivot selection strategy
- Memory-efficient matrix operations
- Numerical stability considerations

## Building and Running

Compile the program using:
```bash
gcc -O3 -DINCLUDE_MAIN -g intopt.c -o intopt -lm
```

Run with an input file:
```bash
./intopt < tests/1/i
```

## Performance Profiling

The code has been optimized using:
- gprof for function-level profiling
- operf for hardware counter analysis
- Valgrind for memory optimization
- Various compiler optimizations (gcc, clang, IBM XL C)

## Requirements

- C compiler (gcc/clang)
- UNIX-like environment
- Math library (-lm)
- Optional: Valgrind, gprof, operf for development
