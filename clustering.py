import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

nsamples = 50
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
random_colors = np.random.rand(nsamples, 3)
tempX = []
tempY = []
ax.scatter(tempX, tempY, s=40, c=random_colors[0])


def plot_init():
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Clustered Radar Points")
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 8)
    ax.grid(True)


def associate_clustering(new_cluster,
                         pre_cluster,
                         max_num_clusters,
                         epsilon,
                         v_factor,
                         use_elevation=False):
    """Associate pre-existing clusters and the new clusters.

    The function performs an association between the pre-existing clusters and the new clusters, with the intent that the
    cluster sizes are filtered.

    Args:
        new_cluster:
        pre_cluster:
        max_num_clusters:
        epsilon:
        v_factor:
        use_elevation:
    """
    num_cluster = max_num_clusters if pre_cluster.shape[
        0] > max_num_clusters else pre_cluster.shape[0]
    pre_avg_vel = np.expand_dims(pre_cluster[num_cluster]['avgVel'], 0)
    pre_location = pre_cluster[num_cluster]['location']

    new_avg_vel = np.expand_dims(new_cluster[num_cluster]['avgVel'], 1)
    new_location = new_cluster[num_cluster]['location']

    # State is previous cluster. output is output cluster.

    # Check if velocity is close.
    # Modify the velocity threshold if the original speed is smaller than threshold itself.
    v_factors = np.ones_like(new_avg_vel) * v_factor
    v_factors = np.minimum(v_factors, new_avg_vel)
    # Put new_cluster as column vector and pre_cluster as row vector to generate difference matrix.
    vel_diff_mat = np.abs(pre_avg_vel - new_avg_vel)
    closest_vel_idx = np.argmin(vel_diff_mat, axis=1)
    closest_vel_val = vel_diff_mat.min(axis=1)

    # Check if position is close enough
    closest_loc = np.zeros_like(len(new_location))
    for i, new_loc in enumerate(new_location):
        loc_diff = (new_loc[0] - pre_location[:, 0]) ** 2 + \
                   (new_loc[1] - pre_location[:, 1]) ** 2 + \
                   (new_loc[2] - pre_location[:, 2]) ** 2 * use_elevation
        closest_loc[i] = np.argmin(loc_diff, axis=1)

    # Get where both velocity and location are satisfied, boolean mask.
    assoc_idx = (closest_vel_val < v_factors) & (closest_loc < epsilon ** 2)
    # Get the actual index. Value j at index i means that pre_cluster[i] is associated to new_cluster[j].
    # if the value j is -1, it means it didn't find any association.
    assoc_idx = (closest_vel_idx + 1) * assoc_idx - 1

    assoc_flag = np.zeros_like(pre_cluster)
    for i, assoc in enumerate(assoc_idx):
        # If there is an associated cluster and not occupied
        if assoc != -1 and not assoc_flag[i]:
            pre_cluster[i] = new_cluster[assoc]
            # IIR filter the size so it won't change rapidly.
            pre_cluster['size'] *= 0.875
        # if this is a new cluster.
        elif assoc != -1:
            np.append(pre_cluster, new_cluster[assoc])
        # if the associated new cluster is occupied.
        else:
            continue

    return pre_cluster


def radar_dbscan(det_obj_2d, weight, doppler_resolution, use_elevation=False):
    """DBSCAN for point cloud. Directly call the scikit-learn.dbscan with customized distance metric.

    DBSCAN algorithm for clustering generated point cloud. It directly calls the dbscan from scikit-learn but with
    customized distance metric to combine the coordinates and weighted velocity information.

    Args:
        det_obj_2d (ndarray): Numpy array containing the rangeIdx, dopplerIdx, peakVal, xyz coordinates of each detected
            points. Can have extra SNR entry, not necessary and not used.
        weight (float): Weight for velocity information in combined distance metric.
        doppler_resolution (float): Granularity of the doppler measurements of the radar.
        use_elevation (bool): Toggle to use elevation information for DBSCAN and output clusters.

    Returns:
        clusters (np.ndarray): Numpy array containing the clusters' information including number of points, center and
            size of the clusters in x,y,z coordinates and average velocity. It is formulated as the structured array for
            numpy.
    """

    # epsilon defines max cluster width
    def custom_distance(obj1, obj2): return \
        (obj1[3] - obj2[3]) ** 2 + \
        (obj1[4] - obj2[4]) ** 2 + \
        use_elevation * (obj1[5] - obj2[5]) ** 2 + \
        weight * ((obj1[1] - obj2[1]) * doppler_resolution) ** 2

    labels = DBSCAN(eps=1.25, min_samples=1,
                    metric=custom_distance).fit_predict(det_obj_2d)
    unique_labels = sorted(
        # Exclude the points clustered as noise, i.e, with negative labels.
        set(labels[labels >= 0]))
    dtype_location = '(' + str(2 + use_elevation) + ',)<f4'
    dtype_clusters = np.dtype({'names': ['num_points', 'center', 'size', 'avgVelocity'],
                               'formats': ['<u4', dtype_location, dtype_location, '<f4']})
    clusters = np.zeros(len(unique_labels), dtype=dtype_clusters)
    for label in unique_labels:
        clusters['num_points'][label] = det_obj_2d[label == labels].shape[0]
        clusters['center'][label] = np.mean(det_obj_2d[label == labels, 3:6], axis=0)[
            :(2 + use_elevation)]
        clusters['size'][label] = np.amax(det_obj_2d[label == labels, 3:6], axis=0)[:(2 + use_elevation)] - \
            np.amin(det_obj_2d[label == labels, 3:6], axis=0)[
            :(2 + use_elevation)]
        clusters['avgVelocity'][label] = np.mean(
            det_obj_2d[label == labels, 1], axis=0) * doppler_resolution

    return clusters


def dbscan_clustering(filteredData, weight=0.8):
    # DBSCAN clustering for point cloud
    X = np.stack((filteredData['x'], filteredData['y'],
                  filteredData['z'], filteredData['velocity']), axis=1)
    # Step 2: Normalize features (VERY IMPORTANT for DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Step 3: Apply DBSCAN
    db = DBSCAN(eps=weight, min_samples=1)  # tune eps based on your data
    labels = db.fit_predict(X_scaled)
    # Step 5: Group points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X[i])
    return clusters


def merge_points_xy(points, x_thresh=0.5, y_thresh=0.5):
    N = len(points)
    visited = np.zeros(N, dtype=bool)
    merged = []

    for i in range(N):
        if visited[i]:
            continue

        # Start a new cluster
        cluster = [points[i]]
        visited[i] = True

        for j in range(i+1, N):
            if visited[j]:
                continue

            # Check your condition
            if abs(points[i][0] - points[j][0]) < x_thresh and \
               abs(points[i][1] - points[j][1]) < y_thresh:
                cluster.append(points[j])
                visited[j] = True

        # Average the cluster
        cluster = np.array(cluster)
        merged.append(cluster.mean(axis=0))

    return np.array(merged)


def plot_update(clusters):
    ax.clear()
    plot_init()
    for i, (label, points) in enumerate(clusters.items()):
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        # creating new scatter chart with updated data
        ax.scatter(x, y, s=40, c=random_colors[i])
    plt.pause(0.1)


# Your data dictionary
test_datas = [{
    'numObj': np.int64(7),
    'x': np.array([0.22087024, -0.07362341,  0.2399578, -2.2332435,  2.891764, 3.2639713, -1.066176], dtype=np.float32),
    'y': np.array([0.75361675, 0.7818577, 0.9293525, 3.2825596, 3.324894, 6.7086244, 1.7002926], dtype=np.float32),
    'z': np.array([0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0., 0., 0., 0., 0., 0., 0.60866284], dtype=np.float32)
}, {
    'numObj': np.int64(6),
    'x': np.array([0.64897674,  0.20859967,  0.2399578, -2.2332435,  2.891764, 3.2639713], dtype=np.float32),
    'y': np.array([0.35906807, 0.71174914, 0.9293525, 3.2825596, 3.324894, 6.7086244], dtype=np.float32),
    'z': np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
}, {
    'numObj': np.int64(4),
    'x': np.array([0.22905062, -2.2332435,  2.891764,  3.2639713], dtype=np.float32),
    'y': np.array([0.8871092, 3.2825596, 3.324894, 6.7086244], dtype=np.float32),
    'z': np.array([0., 0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0., 0., 0., 0.], dtype=np.float32)
}, {
    'numObj': np.int64(3),
    'x': np.array([0.25768194, -2.2332435,  2.891764], dtype=np.float32),
    'y': np.array([0.87921953, 3.2825596, 3.324894], dtype=np.float32),
    'z': np.array([0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0., 0., 0.], dtype=np.float32)
}, {
    'numObj': np.int64(4),
    'x': np.array([-2.2332435,  2.891764,  0.2399578, -0.94074357], dtype=np.float32),
    'y': np.array([3.2825596, 3.324894, 0.9293525, 1.7727741], dtype=np.float32),
    'z': np.array([0., 0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0., 0., 0.12173257, 0.85212797], dtype=np.float32)
}, {'numObj': np.int64(5),
    'x': np.array([0.22905062, -2.2332435,  2.891764,  3.2639713,  0.1567906], dtype=np.float32),
    'y': np.array([0.8871092, 3.2825596, 3.324894, 6.7086244, 0.9911348], dtype=np.float32),
    'z': np.array([0., 0., 0., 0., 0.], dtype=np.float32),
    'velocity': np.array([0.,  0.,  0.,  0., -0.12173257], dtype=np.float32)}
]

for test_data in test_datas:
    clusters = dbscan_clustering(test_data, weight=0.8)

    plot_update(clusters)
    time.sleep(2)

plt.ioff()  # Turn off interactive mode
plt.show()
