import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Neural Network Model
# -------------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# -------------------------------
# PDE Residual
# -------------------------------
def pde_residual(model, x, t, mu):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    # First derivatives
    u_t = torch.autograd.grad(u, t, 
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

    u_x = torch.autograd.grad(u, x,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

    # Second derivative
    u_xx = torch.autograd.grad(u_x, x,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True)[0]

    # PDE residual
    f = u_t - mu * u_xx
    return f

# -------------------------------
# Training Data
# -------------------------------
def generate_training_data(N_f=10000, N_bc=200, N_ic=200):
    
    # Collocation points (inside domain)
    x_f = torch.rand((N_f, 1), device=device)
    t_f = torch.rand((N_f, 1), device=device)

    # Boundary conditions
    t_bc = torch.rand((N_bc, 1), device=device)
    x_bc0 = torch.zeros((N_bc, 1), device=device)
    x_bc1 = torch.ones((N_bc, 1), device=device)

    # Initial condition
    x_ic = torch.rand((N_ic, 1), device=device)
    t_ic = torch.zeros((N_ic, 1), device=device)

    return x_f, t_f, x_bc0, x_bc1, t_bc, x_ic, t_ic

# -------------------------------
# Exact Initial Condition
# -------------------------------
def initial_condition(x):
    return torch.sin(np.pi * x)

# -------------------------------
# Training
# -------------------------------
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

mu = 0.01  # diffusivity

x_f, t_f, x_bc0, x_bc1, t_bc, x_ic, t_ic = generate_training_data()

for epoch in range(5000):

    optimizer.zero_grad()

    # PDE loss
    f = pde_residual(model, x_f, t_f, mu)
    loss_pde = torch.mean(f**2)

    # Boundary loss
    u_bc0 = model(x_bc0, t_bc)
    u_bc1 = model(x_bc1, t_bc)
    loss_bc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

    # Initial condition loss
    u_ic = model(x_ic, t_ic)
    u_ic_true = initial_condition(x_ic)
    loss_ic = torch.mean((u_ic - u_ic_true)**2)

    # Total loss
    loss = loss_pde + 5*loss_bc + loss_ic

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -------------------------------
# Prediction
# -------------------------------
x_test = torch.linspace(0, 1, 100).view(-1,1).to(device)
t_test = torch.ones_like(x_test) * 0.5

u_pred = model(x_test, t_test).detach().cpu().numpy()

print("Prediction done!")


# 1D Plot (u vs x at fixed t)
x = torch.linspace(0, 1, 100).view(-1,1).to(device)
t = torch.ones_like(x) * 0.5  # fixed time

u = model(x, t).detach().cpu().numpy()

plt.figure()
plt.plot(x.cpu().numpy(), u)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("1D Groundwater Flow (t=0.5)")
plt.show()


#2D Plot
# Grid
x = np.linspace(0,1,100)
t = np.linspace(0,1,100)

X, T = np.meshgrid(x, t)

X_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1,1).to(device)
T_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1,1).to(device)

U = model(X_flat, T_flat).detach().cpu().numpy()
U = U.reshape(100,100)

# Heatmap
plt.figure()
plt.contourf(X, T, U, 100)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("2D Heatmap of Groundwater Flow")
plt.show()

#3D Plot

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, T, U, cmap='viridis')

fig.colorbar(surf, ax=ax)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("3D Surface Plot of Groundwater Flow")

plt.show()

# -------------------------------
# Comparison Plot 
# -------------------------------

model.eval()  

x = torch.linspace(0, 1, 100).view(-1,1).to(device)
t_val = 0.5
t = torch.ones_like(x) * t_val

# PINN prediction
u_pred = model(x, t).detach().cpu().numpy().flatten()

# Exact solution
x_np = x.cpu().numpy().flatten()
u_exact = np.sin(np.pi * x_np) * np.exp(-mu * np.pi**2 * t_val)

# Plot
plt.figure(figsize=(8,6))
plt.plot(x_np, u_pred, label="PINN Prediction")
plt.plot(x_np, u_exact, '--', label="Exact Solution")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Comparison Plot at t={t_val}")
plt.legend()
plt.grid()

plt.show()

# -------------------------------
#Error Plot 
# -------------------------------
error = np.abs(u_pred - u_exact)

plt.figure(figsize=(8,6))
plt.plot(x_np, error)

plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("Error Distribution (PINN vs Exact)")
plt.grid()

plt.show()

# -------------------------------
#2D Error Heatmap 
# -------------------------------
error_2D = np.abs(U - (np.sin(np.pi * X) * np.exp(-mu * np.pi**2 * T)))

plt.figure(figsize=(8,6))
plt.contourf(X, T, error_2D, 100)
plt.colorbar()

plt.xlabel("x")
plt.ylabel("t")
plt.title("2D Error Heatmap")

plt.show()

