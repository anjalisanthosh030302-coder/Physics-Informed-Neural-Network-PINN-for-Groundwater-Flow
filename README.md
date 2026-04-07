# Physics-Informed-Neural-Network-PINN-for-Groundwater-Flow


This project implements a **Physics-Informed Neural Network (PINN)** to solve the **1D groundwater flow (diffusion) equation** using PyTorch.

---

## 📌 Problem Statement

We solve the time-dependent diffusion equation:

∂u/∂t = μ ∂²u/∂x²

Where:
- u(x,t): Hydraulic head
- μ: Diffusivity constant

---

## 🧠 Methodology

Instead of traditional numerical methods, this project uses a **Physics-Informed Neural Network (PINN)** that:

- Learns the solution using a neural network
- Enforces:
  - PDE constraints (physics loss)
  - Boundary conditions
  - Initial conditions

---

## ⚙️ Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 🚀 Features

✔ Solves PDE using PINN  
✔ Automatic differentiation for physics constraints  
✔ 1D, 2D, 3D visualizations  
✔ Comparison with analytical solution  
✔ Error analysis (1D + 2D)

---

## 📊 Results

### 🔹 1D Solution
- Smooth curve
- Matches analytical solution

### 🔹 2D Heatmap
- Shows temporal evolution
- Stable diffusion behavior

### 🔹 3D Surface Plot
- Full visualization of u(x,t)

### 🔹 Comparison Plot
- PINN prediction ≈ Exact solution

### 🔹 Error Analysis
- Very low error (~10⁻³)
- Slight increase near boundaries (expected in PINNs)

---

## 📈 Key Insights

- PINNs can accurately solve PDEs with minimal data
- Physics constraints improve generalization
- Suitable for real-world problems in:
  - Hydrology
  - Fluid dynamics
  - Structural engineering

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
