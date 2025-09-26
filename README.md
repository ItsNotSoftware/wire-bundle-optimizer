# 🚀 Wire Bundle Optimizer

A Python app to **pack circular wires** into the **smallest possible circular bundle**.  
It supports multi-start optimization, sleeve layers, and bundle plotting.

---

## Index

1. [Features](#features)
2. [Installation & Run](#installation--run)
3. [UI Walkthrough](#ui-walkthrough)
4. [Predefined Sizes (`wire_types.yaml`)](#predefined-sizes-wire_typesyaml)
5. [Predefined Sleeves (`sleeve_types.yaml`)](#predefined-sleeves-sleeve_typesyaml)
6. [How it Works (Math)](#how-it-works-math)
7. [Tips & Troubleshooting](#tips--troubleshooting)
8. [Screenshots](#screenshots)

---

## Features

-   Define any number of wire types (custom or from **predefined sizes** via YAML).
-   **Color-coded** wires.
-   **Manufacturing margin** (percentage) inflates radii to enforce spacing.
-   **Multi-start SLSQP** with parallel runs to escape local minima.
-   **Inner exclusion** constraint from prior sleeves (no wire can cross a sleeve).
-   Live plot:
    -   Wires placement
    -   Dashed current outer boundary
    -   Colored sleeve rings

---

## Installation & Run

You can run the app in two ways:

### Option 1: Using [uv](https://github.com/astral-sh/uv) (recommended)

```bash
uv run main.py
```

This automatically manages a virtual environment and dependencies.

### Option 2: Using pip

First install dependencies:

```bash
pip install PyQt6 numpy scipy pyyaml
```

Then run:

```bash
python main.py
```

---

## UI Walkthrough

1. **1. Define Wire Types**

    - Set **Count** and **Wire Diameter**:

        - **Custom**: enter diameter in mm.
        - **Predefined Sizes**: pick a label defined in `wire_types.yaml`.

    - Pick a **Color** and click **Add Wire**.
      Identical (color + diameter) entries merge counts automatically.

2. **2. Optimization Parameters**

    - **Number of Solver Initializations**: more restarts → better layout, slower.
    - **Max Solver Iterations**: SLSQP cap per run.
    - **Manufacturing Tolerance Margin**: extra spacing (percent) added to radii.

3. **3. Defined Wires**

    - Shows the current working set.
    - **Remove Selected Wire** or **Clear All** (clears wires, layers, and results).

4. **4. Sleeving**

    - Choose **Custom** thickness or select from **Predefined Sleeves** (from `sleeve_types.yaml`).
    - Pick a **Sleeve color**.
    - Click **Add Sleeve** to add one ring. You can click multiple times to stack multiple sleeves.
    - The first sleeve after an optimize locks the current wire layout as a layer; subsequent sleeves can be added without re-optimizing.
      When clicked:

        - The current optimized bundle becomes a **locked layer** with the specified sleeve thickness and color.
        - The **inner exclusion radius** is updated.
        - The working wire list is cleared for defining the **next ring**.

5. **Optimize and Plot**

    - Runs the multi-start optimizer and updates:

        - Plot (including historic layers)
        - Outer **diameter** in **mm** and **inches**.

6. **Results**

    - The plot is centered and scaled to fit.
    - The window is **scrollable** if content exceeds the viewport.

---

## Predefined Sizes (`wire_types.yaml`)

Place this file next to `main.py`. It maps **labels → diameter (mm)**:

```yaml
# Example — adapt to your catalog
"16 AWG": 1.291
"18 AWG": 1.024
"20 AWG": 0.812
"1.00 mm": 1.000
"0.90 mm": 0.900
"Ethernet": 2.000
```

In the UI, choose **Predefined Sizes** to select one of these keys.

---

## Predefined Sleeves (`sleeve_types.yaml`)

Place this file next to `main.py`. It maps **labels → thickness (mm)**:

```yaml
"Copper mesh (thin)": 0.200
"Copper mesh (heavy)": 0.500
"PET sleeving": 0.500
"PVC jacket": 1.000
```

In the UI, choose **Predefined Sleeves** to select one of these keys. You can also use **Custom** to enter any thickness.

---

## How it Works (Math)

Minimize the enclosing radius $R$ of a circle containing all wire disks.

**Variables**

-   Centers $(x_i, y_i)$ for $i = 1..n$
-   Outer radius $R$

**Effective radii**  
$r_i^{\mathrm{eff}} = r_i \(1 + m)$ where $m$ is the **margin**.

**Constraints**

1. **Containment**:  
   $\|c_i\| + r_i^{\mathrm{eff}} \le R$

2. **Non-overlap**:  
   $\|c_i - c_j\| \ge r_i^{\mathrm{eff}} + r_j^{\mathrm{eff}}$

3. **Frozen core (from sleeves)**:  
   $\|c_i\| \ge R_{\text{core}} + r_i^{\mathrm{eff}}$

**Objective**  
Minimize $R$

---

## Tips & Troubleshooting

-   If optimization is slow, reduce **initializations**.
-   **Add Sleeve** enables after a valid solution or when sleeves already exist.
-   **Clear All** resets everything (wires, layers, results, frozen core).

## Screenshots

### Main UI

This is the main window where you define wires, optimization parameters, and sleeve layers:

![UI Example](ui.png)

---

### Result

After optimization and adding sleeves, the layout will look like this:

![Result Example](result.png)
