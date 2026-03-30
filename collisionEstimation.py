import numpy as np

x_margin = 0.5  # Example margin for collision estimation
y_margin = 1  # Example margin for collision estimation


def radial_to_cartesian_velocity(x, y, vr):
    d = np.sqrt(x**2 + y**2)
    if d == 0:
        return 0, 0
    return -1 * vr * x/d, -1 * vr * y/d


def interval_1d(p0, v, pmin, pmax):
    """Return (t_enter, t_exit) for 1D interval crossing."""
    if abs(v) < 1e-6:
        # No movement in this axis
        if pmin <= p0 <= pmax:
            return -np.inf, np.inf   # always inside
        else:
            return 1, 0              # empty interval (never hits)

    t1 = (pmin - p0) / v
    t2 = (pmax - p0) / v
    return min(t1, t2), max(t1, t2)


def estimateCollisionRadial_V(x, y, vr):
    vx, vy = radial_to_cartesian_velocity(x, y, vr)
    print(f"Cartesian velocity: vx={vx:.2f}, vy={vy:.2f}")

    # Box boundaries
    xmin, xmax = -x_margin, x_margin
    ymin, ymax = -y_margin, y_margin

    tx_in, tx_out = interval_1d(x, vx, xmin, xmax)
    ty_in, ty_out = interval_1d(y, vy, ymin, ymax)

    t_enter = max(tx_in, ty_in)
    t_exit = min(tx_out, ty_out)

    print(f"Collision times: tEnter={t_enter:.2f}, tExit={t_exit:.2f}")

    if t_enter <= t_exit and t_exit >= 0:
        return max(t_enter, 0)

    return None


if __name__ == "__main__":
    # Example usage
    x = 5.0
    y = 5.0
    vr = 3.0
    collision_time = estimateCollisionRadial_V(x, y, vr)
    print(f"Estimated collision time: {collision_time} seconds")
