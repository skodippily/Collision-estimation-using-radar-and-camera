import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import testData as td

# -----------------------------
# Track class
# -----------------------------
colors = np.random.rand(500)


class Track:
    def __init__(self, track_id, x, y):
        self.id = track_id
        self.state = np.array([x, y, 0, 0], dtype=float)  # [x, y, vx, vy]
        self.age = 0
        self.missed = 0

    def predict(self, dt=1.0):
        # Simple constant velocity model
        self.state[0] += self.state[2] * dt
        self.state[1] += self.state[3] * dt

    def update(self, x, y):
        # Simple update (can replace with Kalman)
        vx = x - self.state[0]
        vy = y - self.state[1]

        self.state = np.array([x, y, vx, vy])
        self.missed = 0


# -----------------------------
# Tracker
# -----------------------------
class MultiObjectTracker:
    def __init__(self, dist_threshold=1.0):
        self.tracks = []
        self.next_id = 0
        self.dist_threshold = dist_threshold

    def step(self, detections):
        # Predict all tracks
        for t in self.tracks:
            t.predict()

        if len(self.tracks) == 0:
            for d in detections:
                self._create_track(d)
            return

        # Compute cost matrix (Euclidean distance)
        cost = np.zeros((len(self.tracks), len(detections)))

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                cost[i, j] = np.linalg.norm(t.state[:2] - d)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.dist_threshold:
                self.tracks[r].update(detections[c][0], detections[c][1])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Unmatched tracks → increase missed count
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.missed += 1

        # Delete lost tracks
        self.tracks = [t for t in self.tracks if t.missed < 3]

        # Create new tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in assigned_dets:
                self._create_track(d)

    def _create_track(self, d):
        t = Track(self.next_id, d[0], d[1])
        self.tracks.append(t)
        self.next_id += 1

    def get_tracks(self):
        return [(t.id, t.state.copy()) for t in self.tracks]


def plot_init():
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Clustered Radar Points")
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 8)
    ax.grid(True)


def visualizeTestData(testData):
    indices = [item[0] for item in testData]
    print(f"Test indices####={max(indices)}")
    testData_list = [item[1] for item in testData]

    for frame in testData_list:
        x = np.array([arr[0] for idx, arr in testData])
        y = np.array([arr[1] for idx, arr in testData])

        # Update data instead of re-plotting
        pointsTest = np.column_stack((x, y))
        sc.set_offsets(pointsTest)
        sc.set_array(colors[indices])

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        plt.pause(0.05)


def plot_tracks_2d():
    tracker = MultiObjectTracker()
    for frame in td.Test_radar_data:
        detections = np.column_stack((frame['x'], frame['y']))
        tracker.step(detections)
        visualizeTestData(tracker.get_tracks())


if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create ONE scatter object (important!)
    sc = ax.scatter([], [], s=40, cmap='viridis')
    plot_init()
    plot_tracks_2d()

    plt.ioff()
    plt.show()
