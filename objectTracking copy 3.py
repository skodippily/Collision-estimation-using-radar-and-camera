import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import testData as td
import numpy as np


def compute_cost_matrix(tracks, detections):
    cost = np.zeros((len(tracks), len(detections)))

    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = np.linalg.norm(t - d)  # Euclidean distance

    return cost


def plot_init():
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Clustered Radar Points")
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 8)
    ax.grid(True)


MAX_DIST = 1.0

matches = []
unmatched_tracks = []
unmatched_detections = []


def visualizeTestData(tracks, detections):
    global matches, unmatched_tracks, unmatched_detections
    tracks_xy = tracks[:, :2]
    detections_xy = detections[:, :2]

    sc_tracks.set_offsets(tracks_xy)
    sc_dets.set_offsets(detections_xy)

    # Remove old lines
    for line in ax.lines:
        line.remove()

    # Draw new match lines
    for t_idx, d_idx in matches:
        if t_idx < len(tracks_xy) and d_idx < len(detections_xy):
            tx, ty = tracks_xy[t_idx]
            dx, dy = detections_xy[d_idx]
            ax.plot([tx, dx], [ty, dy], linewidth=1.2)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.05)


def identify_matches(tracks, detections):
    global matches, unmatched_tracks, unmatched_detections
    cost = compute_cost_matrix(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost)

    print("Assignments:")
    for r, c in zip(row_ind, col_ind):
        print(f"Track {r} → Detection {c}, distance={cost[r, c]:.2f}")

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < MAX_DIST:
            matches.append((r, c))
        else:
            unmatched_tracks.append(r)
            unmatched_detections.append(c)

    # Add remaining unmatched
    for i in range(len(tracks)):
        if i not in row_ind:
            unmatched_tracks.append(i)

    for j in range(len(detections)):
        if j not in col_ind:
            unmatched_detections.append(j)


def run_test(Test_radar_data):
    global matches, unmatched_tracks, unmatched_detections
    previous_frame = None
    for frame in Test_radar_data:
        print(f"Processing with {frame['numObj']} objects")
        if previous_frame is None:
            previous_frame = frame
            continue

        matches.clear()
        unmatched_tracks.clear()
        unmatched_detections.clear()

        tracks = np.array(list(zip(previous_frame['x'], previous_frame['y'])))
        detections = np.array(
            list(zip(frame['x'], frame['y'])))
        cost = compute_cost_matrix(tracks, detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        print("Assignments:")
        for r, c in zip(row_ind, col_ind):
            print(f"Track {r} → Detection {c}, distance={cost[r, c]:.2f}")

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < MAX_DIST:
                matches.append((r, c))
            else:
                unmatched_tracks.append(r)
                unmatched_detections.append(c)

        # Add remaining unmatched
        for i in range(len(tracks)):
            if i not in row_ind:
                unmatched_tracks.append(i)

        for j in range(len(detections)):
            if j not in col_ind:
                unmatched_detections.append(j)

        print(f"no of Matches: {len(matches)} ot of {len(detections)}")
        print("Unmatched Tracks:", unmatched_tracks)
        print("Unmatched Detections:", unmatched_detections)
        # Here you would call your tracking update function
        # For example: tracker.update(frame)
        visualizeTestData(tracks, detections)
        previous_frame = frame


if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    sc_tracks = ax.scatter([], [], c='blue', s=40, label='Tracks')
    sc_dets = ax.scatter([], [], c='red',  s=40, label='Detections')

    # Create ONE scatter object (important!)
    sc = ax.scatter([], [], s=40)
    plot_init()
    run_test(td.Test_radar_data)

    plt.ioff()
    plt.show()
