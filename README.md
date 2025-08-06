Sure! Here’s the exact markdown text you can put in your README file to get the polished look I formatted above:

````markdown
# Astrophysical SPH Simulation Engine — User Guide

Welcome to the **Astrophysical SPH Simulation Engine**!  
This engine allows you to generate initial conditions and run astrophysical Smoothed Particle Hydrodynamics (SPH) simulations through a simple Command Line Interface (CLI)-like experience using Julia’s `ArgParse` package.

---

## Getting Started

### 1. Clone the Git Repository

```bash
git clone <repository_url>
cd fastv1_kd&single_oc
````

### 2. Launch Julia REPL

```bash
julia
```

### 3. Load the ArgParse Package and Set Arguments

Inside the Julia REPL, you will configure the arguments (`ARGS` variable) to control your workflow.

---

## Available Modes

The engine supports two base options controlled by the `ARGS` variable:

### a) Generate Initial Conditions (`--generate`)

* Creates a CSV file with initial conditions based on predefined functions in `iniconds.jl`.
* All constants and parameters (e.g., gas temperature) can be adjusted **only** inside `iniconds.jl`.
* This design keeps the CLI arguments simple and clean.

### b) Run Simulations (`--run`)

* Runs simulations for gases with either a **Polytropic EOS** or **Isothermal EOS**.
* Due to simplifications in isothermal gases, there are separate simulation files optimized for each EOS type.
* Simulation parameters include snapshot controls and plotting options.

---

## Examples

### Example 1: Generate Initial Conditions for Gaussian Sphere with Polytropic EOS

```julia
ARGS = ["--generate", "--EOS", "polytropic", "--ic-type", "gaussian_sphere"]
include("sph_manager.jl")
```

* This generates `1snap.csv` inside the folder named after the IC type (`gaussian_sphere`).

---

### Example 2: Run Simulation Using the Above Initial Conditions

```julia
ARGS = ["--run", "--EOS", "polytropic", "--ic-type", "gaussian_sphere", "1", "5", "true", "false"]
include("sph_manager.jl")
```

* This starts a polytropic simulation using snapshot ID `1`.
* Snapshots will be taken every 5 steps (`snapshot interval = 5`).
* The `true` flag means snapshots are saved.
* The `false` flag disables saving of PNG plots.

---

## Notes

* Modify any physical parameters or constants only in `iniconds.jl` to maintain consistency.
* The CLI interface (`ARGS`) keeps terminal commands concise.
* Refer to comments inside `iniconds.jl` and `sph_manager.jl` for detailed customization options.

---

If you have questions or want to contribute, feel free to open an issue or submit a pull request!

```

---

If you paste this whole block into your README.md file, it will render nicely on GitHub or any markdown viewer. Let me know if you want it tailored for other formats!
```
