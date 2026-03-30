import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------
# EKF TRACK CLASS
# ---------------------------


class EKFTrack:
    def __init__(self, z, track_id):
        self.id = track_id

        r, theta, vr = z

        # Initialize position from polar
        px = r * np.cos(theta)
        py = r * np.sin(theta)

        # Initialize velocity (approx)
        vx = vr * np.cos(theta)
        vy = vr * np.sin(theta)

        self.x = np.array([px, py, vx, vy])

        self.P = np.eye(4) * 10  # covariance
        self.Q = np.eye(4) * 0.1  # process noise
        self.R = np.diag([1.0, 0.05, 1.0])  # measurement noise

        self.missed = 0

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def h(self, x):
        px, py, vx, vy = x

        r = np.sqrt(px**2 + py**2)
        theta = np.arctan2(py, px)
        vr = (px*vx + py*vy) / (r + 1e-6)

        return np.array([r, theta, vr])

    def jacobian(self, x):
        px, py, vx, vy = x
        r = np.sqrt(px**2 + py**2) + 1e-6

        H = np.zeros((3, 4))

        # dr/dx
        H[0, 0] = px / r
        H[0, 1] = py / r

        # dtheta/dx
        H[1, 0] = -py / (r**2)
        H[1, 1] = px / (r**2)

        # dvr/dx
        H[2, 0] = (vx*r - px*(px*vx + py*vy)/r) / (r**2)
        H[2, 1] = (vy*r - py*(px*vx + py*vy)/r) / (r**2)
        H[2, 2] = px / r
        H[2, 3] = py / r

        return H

    def update(self, z):
        z_pred = self.h(self.x)
        H = self.jacobian(self.x)

        y = z - z_pred

        # Normalize angle
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.missed = 0

    def mahalanobis(self, z):
        z_pred = self.h(self.x)
        H = self.jacobian(self.x)

        y = z - z_pred
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        S = H @ self.P @ H.T + self.R
        return y.T @ np.linalg.inv(S) @ y


# ---------------------------
# TRACKER CLASS
# ---------------------------
class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.max_missed = 5
        self.gate_threshold = 9.21  # Chi-square (3 DOF)

    def update(self, detections, dt):
        # 1. Predict all tracks
        for track in self.tracks:
            track.predict(dt)

        if len(self.tracks) == 0:
            for z in detections:
                self.tracks.append(EKFTrack(z, self.next_id))
                self.next_id += 1
            return

        # 2. Cost matrix (Mahalanobis distance)
        cost = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            for j, z in enumerate(detections):
                d = track.mahalanobis(z)
                cost[i, j] = d if d < self.gate_threshold else 1e6

        # 3. Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_detections = set()

        # 4. Update assigned tracks
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.gate_threshold:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # 5. Handle unassigned tracks
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track.missed += 1

        # 6. Delete lost tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        # 7. Create new tracks
        for i, z in enumerate(detections):
            if i not in assigned_detections:
                self.tracks.append(EKFTrack(z, self.next_id))
                self.next_id += 1


def plot_tracks_2d(tracker):
    plt.clf()

    # Plot radar origin
    plt.scatter(0, 0, marker='x')
    plt.text(0, 0, "Radar")

    for track in tracker.tracks:
        px, py, vx, vy = track.x

        # Position
        plt.scatter(px, py)

        # Track ID
        plt.text(px, py, f"ID {track.id}")

        # Velocity arrow
        plt.arrow(px, py, vx, vy, head_width=0.3, length_includes_head=True)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Radar Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.01)


# ---------------------------
# EXAMPLE USAGE
# ---------------------------
if __name__ == "__main__":
    tracker = MultiObjectTracker()

    # Example detections: [range, angle(rad), radial velocity]
    detections = [
        np.array([10, 0.5, 2]),
        np.array([15, -0.2, -1])
    ]

    dt = 0.1  # time step

    for t in range(100):
        if t == 10:
            detections.append(np.array([20, 0.1, 3]))  # new object appears
        elif t == 20:
            detections.pop(1)  # second object disappears
        tracker.update(detections, dt)

        plot_tracks_2d(tracker)
        print(f"Time {t}")
        for track in tracker.tracks:
            print(f"ID {track.id}: {track.x}")
