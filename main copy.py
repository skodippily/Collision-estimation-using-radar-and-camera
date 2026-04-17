import threading
import time
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from main_testcode import getClosestCluster
import clustering as cluster
import objectTracking as ot
import collisionEstimation as ce

from AWR1843_Read_Data import readData_AWR1843 as radar

start_time = 0
angle_of_front = 45
data_structure = []


class RadarReading:
    def __init__(self):
        # Create a stop flag for safe shutdown
        self.stop_event = threading.Event()
        self.start_time = start_time
        self.angle_of_front = angle_of_front
        self.data_structure = data_structure

    def upateRadarData(self):
        while not self.stop_event.is_set():
            radar.updateFromMain()
            time.sleep(0.1)

    def dataProcess(self):
        while not self.stop_event.is_set():
            print(f"Test dict####={radar.getData()}")
            time.sleep(0.05)

    def measure_time(self, startbit):
        if startbit:
            self.start_time = time.time()
            return self.start_time
        return time.time() - self.start_time

    # measure_time(startbit=True)  # Start timing
    # measure_time(startbit=False)

    def visualize_clusters(self, ax, clusters, matched_pairs=[]):
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
        colors = mpl.colormaps['tab10'].resampled(max(num_clusters, 1))

        for label, points in clusters.items():
            # print(f"Cluster label: {label}, Points: {points}")

            if label == -1:
                # Noise points
                color = 'black'
                label_name = 'Noise'
            else:
                color = colors(label)
                label_name = f'Cluster {label}'

            if np.array(points).ndim == 2:
                x = np.array([a[0] for a in points])
                y = np.array([a[1] for a in points])
            else:
                x = points[0]
                y = points[1]

            ax.scatter(x, y,
                       c=[color],
                       label=label_name,
                       s=40)
        if matched_pairs is not None and matched_pairs != []:
            for (x1, y1), (x2, y2) in matched_pairs:
                ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.2)
                ax.scatter(x2, y2, marker='v', s=40)

    def getClosestCluster(self, points):
        """Get the closest cluster to the origin."""
        min_distance = float('inf')
        closest_cluster = None
        for point in points:
            distance = np.sqrt(point[0]**2 + point[1]**2)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = point
        return closest_cluster

    def clean_clusters(self, clusters, remove_noise=True):
        """Remove noise points from clusters."""
        cleaned_clusters = {}
        for label, points in clusters.items():
            if label == -1:
                if not remove_noise:
                    # cleaned_clusters[label] = self.getClosestCluster(points)
                    cleaned_clusters[label] = points
                continue
            cleaned_clusters[label] = self.getClosestCluster(points)
        return cleaned_clusters

    def clusters_reform(self, clusters):
        """Reform clusters into a more structured format."""
        reformatted_clusters = []
        for label, points in clusters.items():
            _points = points.copy()
            if (np.array(_points).ndim == 1):
                # Remove z-axis if present
                reformatted_clusters.append(np.delete(_points, 2))
            else:
                for point in _points:
                    reformatted_clusters.append(np.delete(point, 2))
        return reformatted_clusters

    def get_matched_pairs(self, current_clusters, previous_clusters):
        """Get matched pairs of tracks and clusters."""
        matched_pairs = []
        if previous_clusters == []:
            return None
        # match current clusters to previous clusters based on closest distance
        for currID, currData in current_clusters.items():
            for prevID, prevData in previous_clusters.items():
                if currID == prevID:
                    matched_pairs.append(
                        [(prevData[0], prevData[1]), (currData[0], currData[1])])
                    break
        return matched_pairs

    def get_direction(self, x, y):
        """Get the direction of the cluster based on its position."""
        theta = np.degrees(np.arctan2(x, y))
        if abs(theta) < self.angle_of_front:
            return "front"
        if self.angle_of_front >= theta:
            return "right"
        if self.angle_of_front <= theta:
            return "left"

    def main(self):
        radar.initRadar(True)
        time.sleep(2)
        # Create threads
        radarThread = threading.Thread(target=self.upateRadarData)
        processingThread = threading.Thread(target=self.dataProcess)

        # Start threads
        # radarThread.start()
        # processingThread.start()

        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        tracker = ot.MultiObjectTracker()
        previous_frame = []

        print("Threads started. Press Ctrl+C to stop.")

        # Keep main thread alive
        try:
            while not self.stop_event.is_set():
                radar.updateFromMain()
                print(f"Test dict####={radar.getData()}")
                radar.updatePlot()
                plt.pause(0.1)
                # print(f"identified_clusters: {identified_clusters}")

            plt.ioff()
            plt.show()

        except KeyboardInterrupt:
            print("\nStopping threads...")
            self.stop_event.set()

        # Wait for threads to finish
        radarThread.join()
        processingThread.join()
        print("All threads closed safely.")


if __name__ == "__main__":
    try:
        rr = RadarReading()
        rr.main()
        # radar.test()
        # AWR1843 = AWR1843_Read_Data
        # AWR1843.test()
        pass

    except KeyboardInterrupt:
        rr.stop_event.set()
        radar.closePortsAndPlot()
        print("\nClosing...")
