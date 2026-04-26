import threading
import time
import numpy as np
import cv2
import uvicorn

import matplotlib.pyplot as plt
import matplotlib as mpl
import testData as td
import clustering as cluster
import objectTracking as ot
import collisionEstimation as ce
import shared_state

from AWR1843_Read_Data import readData_AWR1843 as radar
from Object_detection import objectDetection as od


class RadarReading:
    def __init__(self,
                 start_time=0,
                 angle_of_front=45,
                 model="Object_detection/yolov8n.pt",
                 source="JETSON"):
        # Create a stop flag for safe shutdown
        self.stop_event = threading.Event()
        self.start_time = start_time
        self.angle_of_front = angle_of_front
        self.data_structure = []
        self.model = model
        self.source = source

    def upateRadarData(self):
        while not self.stop_event.is_set():
            radar.updateFromMain()
            self.radar_data = radar.getData()
            # radar.updatePlot()

    def runObjectDetection(self, odtracker):
        odtracker.run()

    def measure_time(self, startbit):
        if startbit:
            self.start_time = time.time()
            return self.start_time
        return time.time() - self.start_time

    # measure_time(startbit=True)  # Start timing
    # measure_time(startbit=False)
    def init_plot(self, ax):
        ax.set_title("DBSCAN Clustering Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 8)
        ax.grid(True)

        self.scatter_plots = {}   # store scatter objects
        self.line_plots = []      # store line objects

    def visualize_clusters(self, ax, clusters, matched_pairs=[]):
        """
        Visualize clustered points with different colors.
        """

        num_clusters = len(clusters)
        colors = mpl.colormaps['tab10'].resampled(max(num_clusters, 1))

        # ---- Update scatter plots ----
        for label, points in clusters.items():

            if label == -1:
                color = 'black'
            else:
                color = colors(label)

            points = np.array(points)

            if points.ndim == 2:
                xy = points[:, :2]
            else:
                xy = np.array([[points[0], points[1]]])

            if label not in self.scatter_plots:
                # Create once
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=[color], s=40)
                self.scatter_plots[label] = sc
            else:
                # Update existing scatter
                self.scatter_plots[label].set_offsets(xy)

        # ---- Remove unused clusters ----
        existing_labels = set(self.scatter_plots.keys())
        new_labels = set(clusters.keys())

        for label in existing_labels - new_labels:
            self.scatter_plots[label].remove()
            del self.scatter_plots[label]

        # ---- Update lines ----
        for line in self.line_plots:
            line.remove()
        self.line_plots.clear()

        if matched_pairs:
            for (x1, y1), (x2, y2) in matched_pairs:
                line, = ax.plot([x1, x2], [y1, y2],
                                color='green', linewidth=1.2)
                self.line_plots.append(line)

        plt.pause(0.1)

    def crop_radar_data(self, radar_data, z_range=(-1, 1), flatten=False):
        # Keep only points inside z range
        mask = (
            (radar_data['z'] >= z_range[0]) &
            (radar_data['z'] <= z_range[1])
        )

        radar_data = {
            'x': radar_data['x'][mask],
            'y': radar_data['y'][mask],
            'z': radar_data['z'][mask],
            'velocity': radar_data['velocity'][mask],
        }
        # Update number of objects
        radar_data['numObj'] = np.int64(np.sum(mask))

        # Optional flatten
        if flatten:
            radar_data['z'] = np.zeros_like(radar_data['z'])

        return radar_data

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
                    # cleaned_clusters[label] = getClosestCluster(points)
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
        radar.initRadar()
        self.odtracker = od.YOLOTracker(
            model_path="yolov8n.pt",
            source=0,  # or "JETSON"
            confidence=0.7
        )

        # Start FastAPI server in a background thread
        import os
        os.environ["REAL_DATA_MODE"] = "1"
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=("app:app",),
            kwargs={"host": "0.0.0.0", "port": 8000},
            daemon=True
        )
        server_thread.start()

        time.sleep(2)

        plt.ion()
        plt.rcParams['figure.raise_window'] = False
        fig, ax = plt.subplots(figsize=(6, 6))
        self.init_plot(ax)
        plt.show()
        tracker = ot.MultiObjectTracker()
        previous_frame = []

        # Create threads
        radarThread = threading.Thread(target=self.upateRadarData, daemon=True)
        objectDetection = threading.Thread(
            target=self.runObjectDetection, args=(self.odtracker,), daemon=True)

        radar.updateFromMain()
        self.radar_data = radar.getData()
        # Start threads
        radarThread.start()
        objectDetection.start()
        time.sleep(5)

        # Keep main thread alive
        try:
            while not self.stop_event.is_set():
                # Radar data processing...
                # Perform clustering
                if self.radar_data is None or len(self.radar_data['x']) == 0:
                    continue
                self.measure_time(startbit=True)  # Start timing

                cropped_radar = self.crop_radar_data(self.radar_data)
                if cropped_radar['numObj'] == 0:
                    continue
                clusters = cluster.dbscan_clustering(
                    cropped_radar, weight=0.5)

                # Clean clusters
                cleaned_clusters = self.clean_clusters(
                    clusters, remove_noise=False)

                # identify matches
                reform_clusters = self.clusters_reform(cleaned_clusters)
                identified_clusters = ot.identify_clusters(
                    tracker, reform_clusters)
                matched_pairs = self.get_matched_pairs(
                    identified_clusters, previous_frame)
                previous_frame = identified_clusters
                self.visualize_clusters(
                    ax, cleaned_clusters, matched_pairs=matched_pairs)

                shared_state.data_structure = []
                for label, point in identified_clusters.items():
                    collision_time = ce.estimateCollision(
                        point[0], point[1], vx=point[2], vy=point[3])
                    # print(
                    #     f"Estimated collision time for cluster {label}: {collision_time} seconds")
                    shared_state.data_structure.append({
                        'id': label,
                        'object': 'unknown',
                        # Estimate distance from the origin
                        'distance': np.sqrt(point[0]**2 + point[1]**2),
                        # Estimate speed from velocity components
                        'speed': np.sqrt(point[2]**2 + point[3]**2),
                        'direction': self.get_direction(point[0], point[1]),
                        'ttc': collision_time
                    })

                clean_time = self.measure_time(startbit=False)
                print(
                    f"Tracking algorithm takes {clean_time:.2f} seconds to run.")

                # Camera data processing...
                results = self.odtracker.getResults()
                if results is not None:
                    for box in results.boxes:
                        print(
                            f"Detected object: {results.names[int(box.cls[0])]} with confidence {box.conf[0]:.2f}")
        except KeyboardInterrupt:
            print("\nStopping threads...")
            self.stop_event.set()

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
