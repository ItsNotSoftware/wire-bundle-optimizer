# 🚀 Wire Bundle Optimizer

A Python-based optimization tool to tightly arrange circular wires of varying radii within the smallest possible circular cross-section — **minimizing diameter** while ensuring **no overlaps**.

This project was inspired by a real-world engineering challenge at **Rocket Factory Augsburg**: routing hundreds of wires through a constrained tunnel inside a **rocket fuel tank**. 

The tool supports:
- Any number of wires with arbitrary radii
- Multiple randomized restarts for global search
- Parallel execution for faster optimization
- Visual plot of the optimized layout

## Mathematical Formulation

The optimization problem is defined as follows:

- **Objective:**  
  Minimize the **outer radius** `R` of a circle enclosing all wires.

- **Decision Variables:**  
  - **Positions of the wire centers:**  
    Each wire has a center at coordinates `(x_i, y_i)` for `i = 1, ..., n`

  - **Outer radius of the bundle:**  
    Denoted as `R`

- **Constraints:**  
  1. **Containment:** Each wire must lie entirely within the outer circle:  
     `sqrt(x_i^2 + y_i^2) + r_i <= R` for all `i`  
     
  2. **Non-overlap:** No two wires may intersect:  
     `sqrt((x_i - x_j)^2 + (y_i - y_j)^2) >= r_i + r_j` for all `i < j`

- **Optimization Method:**  
  The solver uses Sequential Least Squares Programming (**SLSQP**), a gradient-based constrained optimizer. Initial wire positions are randomly generated across multiple runs to escape local minima.

- **Output:**  
  - Optimized coordinates `(x_i, y_i)` for each wire  
  - Final minimized outer **diameter** (`2 * R`)  
  - 2D plot of the wire layout and enclosing circle

## 🚀 How to Run

To run the optimizer, you have two options:

### Option 1: Using `uv` (recommended)

If you have [`uv`](https://github.com/astral-sh/uv) installed:

```bash
uv run main.py
````

This will automatically set up a virtual environment and install all dependencies.

### Option 2: Manual Setup with `pip`

If you're not using `uv`, you can install the dependencies manually. Look inside `pyproject.toml` and install the required packages with:

```bash
pip install <dependency1> <dependency2> ...
```

Then run the optimizer:

```bash
python main.py
```
