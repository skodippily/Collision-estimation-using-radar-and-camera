import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment

dt = 1.0
DIST_THRESHOLD = 2.0
MAX_MISSES = 3

# -------------------------------
# UKF Model
# -------------------------------


def fx(x, dt):
    px, py, vx, vy = x
    return np.array([px + vx*dt, py + vy*dt, vx, vy])


def hx(x):
    px, py, vx, vy = x
    r = np.sqrt(px**2 + py**2)
    vr = (px*vx + py*vy)/r if r > 1e-6 else 0.0
    return np.array([px, py, vr])


def create_ukf(init_meas):
    points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(
        dim_x=4, dim_z=3, dt=dt, fx=fx, hx=hx, points=points)

    ukf.x = np.array([init_meas[0], init_meas[1], 0., 0.])
    ukf.P *= 10
    ukf.Q = np.eye(4) * 0.1
    ukf.R = np.diag([0.3, 0.3, 0.2])

    return ukf

# -------------------------------
# Track class
# -------------------------------


class Track:
    def __init__(self, id, measurement):
        self.id = id
        self.ukf = create_ukf(measurement)
        self.misses = 0

    def predict(self):
        self.ukf.predict()

    def update(self, measurement):
        self.ukf.update(measurement)
        self.misses = 0

    def miss(self):
        self.misses += 1

    def get_state(self):
        return self.ukf.x

# -------------------------------
# Tracker
# -------------------------------


class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def step(self, detections):
        """
        detections: list of [x, y, vr]
        """

        # 1. Predict all tracks
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
            return

        # 2. Build cost matrix (Euclidean distance)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            tx, ty = track.get_state()[0:2]
            for j, det in enumerate(detections):
                dx, dy = det[0], det[1]
                cost_matrix[i, j] = np.linalg.norm([tx - dx, ty - dy])

        # 3. Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        # 4. Update matched
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < DIST_THRESHOLD:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # 5. Handle unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track.miss()

        # 6. Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses < MAX_MISSES]

        # 7. Create new tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1

    def get_tracks(self):
        _temp = {}
        for t in self.tracks:
            x, y, vx, vy = t.get_state()
            # print(f"ID {t.id}: x={x:.2f}, y={y:.2f}, vx={vx:.2f}, vy={vy:.2f}")
            _temp[t.id] = [x, y, vx, vy]
        return _temp


def identify_clusters(tracker: MultiObjectTracker, radar_data):
    """Identify clusters in radar data."""
    tracker.step(radar_data)
    return tracker.get_tracks()


if __name__ == "__main__":
    tracker = MultiObjectTracker()

    frames = [
        [[1, 1, 1.0], [5, 5, -1.0]],
        [[2, 1.5, 1.1], [4.5, 5.2, -1.1]],
        [[3, 2, 1.2], [4.0, 5.5, -1.2]],
    ]

    for i, detections in enumerate(frames):
        print(f"\nFrame {i+1}")
        tracker.step(detections)
        tracker.print_tracks()
