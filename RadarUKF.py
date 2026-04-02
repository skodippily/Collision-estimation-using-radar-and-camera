import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

dt = 1.0

# -------------------------------
# 1. State transition (same)
# -------------------------------


def fx(x, dt):
    px, py, vx, vy = x
    return np.array([
        px + vx * dt,
        py + vy * dt,
        vx,
        vy
    ])

# -------------------------------
# 2. Measurement function (UPDATED)
# z = [x, y, vr]
# -------------------------------


def hx(x):
    px, py, vx, vy = x

    # radial velocity
    r = np.sqrt(px**2 + py**2)
    if r < 1e-6:
        vr = 0.0
    else:
        vr = (px * vx + py * vy) / r

    return np.array([px, py, vr])

# -------------------------------
# 3. Residual function (optional)
# -------------------------------


def residual_z(a, b):
    return a - b  # no angle now → simple subtraction


# -------------------------------
# 4. Sigma points
# -------------------------------
points = MerweScaledSigmaPoints(
    n=4, alpha=0.1, beta=2.0, kappa=0
)

# -------------------------------
# 5. Create UKF
# -------------------------------
ukf = UnscentedKalmanFilter(
    dim_x=4,
    dim_z=3,   # [x, y, vr]
    dt=dt,
    fx=fx,
    hx=hx,
    points=points
)

ukf.residual_z = residual_z

# Initial state
ukf.x = np.array([0., 0., 1., 1.])

# Covariance
ukf.P *= 10

# Process noise
ukf.Q = np.eye(4) * 0.1

# Measurement noise
ukf.R = np.diag([
    0.3,   # x noise
    0.3,   # y noise
    0.2    # vr noise
])

# -------------------------------
# 6. Dummy radar measurements
# Format: [x, y, z, vr]
# (z ignored)
# -------------------------------
measurements = [
    [1.0, 0.5, 0.0, 1.2],
    [2.0, 1.0, 0.0, 1.1],
    [3.1, 1.5, 0.0, 1.3],
    [4.0, 2.1, 0.0, 1.2],
    [5.2, 2.5, 0.0, 1.25]
]

# -------------------------------
# 7. Run UKF
# -------------------------------
for i, z in enumerate(measurements):
    z = np.array([z[0], z[1], z[3]])  # ignore z-axis

    ukf.predict()
    ukf.update(z)

    print(f"Step {i+1}")
    print(f"x = {ukf.x[0]:.2f}, y = {ukf.x[1]:.2f}")
    print(f"vx = {ukf.x[2]:.2f}, vy = {ukf.x[3]:.2f}")
    print("-" * 30)
