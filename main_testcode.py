"""Test code to check the clustering of radar and camera data using DBSCAN and point association."""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import testData as td
import clustering as cluster
import objectTracking as ot
import collisionEstimation as ce


start_time = 0
angle_of_front = 45
data_structure = []


def measure_time(startbit):
    global start_time
    if startbit:
        start_time = time.time()
        return start_time
    return time.time() - start_time


# measure_time(startbit=True)  # Start timing
# measure_time(startbit=False)


def visualize_clusters(ax, clusters, matched_pairs=[]):
    """
    Visualize clustered points with different colors.
    """

    ax.clear()

    ax.set_title("DBSCAN Clustering Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 8)
    ax.grid(True)

    num_clusters = len(clusters)
    colors = mpl.colormaps["tab10"].resampled(max(num_clusters, 1))

    for label, points in clusters.items():
        # print(f"Cluster label: {label}, Points: {points}")

        if label == -1:
            # Noise points
            color = "black"
            label_name = "Noise"
        else:
            color = colors(label)
            label_name = f"Cluster {label}"

        if np.array(points).ndim == 2:
            x = np.array([a[0] for a in points])
            y = np.array([a[1] for a in points])
        else:
            x = points[0]
            y = points[1]

        ax.scatter(x, y, c=[color], label=label_name, s=40)
    if matched_pairs is not None and matched_pairs != []:
        for (x1, y1), (x2, y2) in matched_pairs:
            ax.plot([x1, x2], [y1, y2], color="green", linewidth=1.2)
            ax.scatter(x2, y2, marker="v", s=40)


def getClosestCluster(points):
    """Get the closest cluster to the origin."""
    min_distance = float("inf")
    closest_cluster = None
    for point in points:
        distance = np.sqrt(point[0] ** 2 + point[1] ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = point
    return closest_cluster


def clean_clusters(clusters, remove_noise=True):
    """Remove noise points from clusters."""
    cleaned_clusters = {}
    for label, points in clusters.items():
        if label == -1:
            if not remove_noise:
                # cleaned_clusters[label] = getClosestCluster(points)
                cleaned_clusters[label] = points
            continue
        cleaned_clusters[label] = getClosestCluster(points)
    return cleaned_clusters


def clusters_reform(clusters):
    """Reform clusters into a more structured format."""
    reformatted_clusters = []
    for label, points in clusters.items():
        _points = points.copy()
        if np.array(_points).ndim == 1:
            # Remove z-axis if present
            reformatted_clusters.append(np.delete(_points, 2))
        else:
            for point in _points:
                reformatted_clusters.append(np.delete(point, 2))
    return reformatted_clusters


def get_matched_pairs(current_clusters, previous_clusters):
    """Get matched pairs of tracks and clusters."""
    matched_pairs = []
    if previous_clusters == []:
        return None
    # match current clusters to previous clusters based on closest distance
    for currID, currData in current_clusters.items():
        for prevID, prevData in previous_clusters.items():
            if currID == prevID:
                matched_pairs.append(
                    [(prevData[0], prevData[1]), (currData[0], currData[1])]
                )
                break
    return matched_pairs


def get_direction(x, y):
    """Get the direction of the cluster based on its position."""
    theta = np.degrees(np.arctan2(x, y))
    if abs(theta) < angle_of_front:
        return "front"
    if angle_of_front >= theta:
        return "right"
    if angle_of_front <= theta:
        return "left"


def simulate_radar_data(Test_radar_data):
    """Simulate radar data for testing (NO visualization)."""

    global data_structure

    tracker = ot.MultiObjectTracker()
    previous_frame = []

    for radar_data in np.array(Test_radar_data):

        # --- Clustering ---
        measure_time(startbit=True)
        clusters = cluster.dbscan_clustering(radar_data, weight=0.8)
        cluster_time = measure_time(startbit=False)
        print(f"Clustering algorithm takes {cluster_time:.2f} seconds to run.")

        # --- Cleaning ---
        measure_time(startbit=True)
        cleaned_clusters = clean_clusters(clusters, remove_noise=False)
        clean_time = measure_time(startbit=False)
        print(f"Cleaning algorithm takes {clean_time:.2f} seconds to run.")

        # --- Tracking ---
        measure_time(startbit=True)
        reform_clusters = clusters_reform(cleaned_clusters)
        identified_clusters = ot.identify_clusters(tracker, reform_clusters)
        previous_frame = identified_clusters
        track_time = measure_time(startbit=False)
        print(f"Tracking algorithm takes {track_time:.2f} seconds to run.")

        # --- Collision estimation ---
        data_structure.clear()

        for label, point in identified_clusters.items():
            collision_time = ce.estimateCollision(
                point[0], point[1], vx=point[2], vy=point[3]
            )

            print(
                f"Estimated collision time for cluster {label}: {collision_time} seconds"
            )

            data_structure.append(
                {
                    "id": int(label),
                    "object": "unknown",
                    "distance": float(np.sqrt(point[0] ** 2 + point[1] ** 2)),
                    "speed": float(np.sqrt(point[2] ** 2 + point[3] ** 2)),
                    "direction": get_direction(point[0], point[1]),
                    "ttc": (
                        float(collision_time) if collision_time is not None else None
                    ),
                }
            )

        print(data_structure)

        time.sleep(0.15)


if __name__ == "__main__":
    simulate_radar_data(td.Test_radar_data)
