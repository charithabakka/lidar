import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import DBSCAN

# ---------- 1. Load Data ----------
# Define file paths
camera_image_path = "path/to/your/image.png"
lidar_data_path = "path/to/your/pointcloud.npz"

# Load camera image
image = cv2.imread(camera_image_path)
if image is None:
    raise ValueError("Camera image not found!")

# Load LiDAR data from NPZ file
with np.load(lidar_data_path) as data:
    lidar_points = data['arr_0']
print(f"LiDAR Data Shape: {lidar_points.shape}")

# ---------- 2. Preprocess LiDAR Data ----------
# Filter LiDAR points (e.g., within 50 meters and above ground level)
filtered_points = lidar_points[
    (lidar_points[:, 0] > 0) & (lidar_points[:, 0] < 50) & (lidar_points[:, 2] > -2)
]

# ---------- 3. Obstacle Detection using DBSCAN ----------
# Use X, Y coordinates for clustering
clustering = DBSCAN(eps=0.8, min_samples=10).fit(filtered_points[:, :2])
labels = clustering.labels_

# Extract clustered (non-noise) obstacle points
obstacle_points = filtered_points[labels != -1]

# ---------- 4. Projection Function ----------
# Example Intrinsic Camera Parameters (replace with real calibration)
fx, fy = 1000, 1000  # Focal length
cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point

def project_point(point):
    """Project a 3D point onto the 2D image plane."""
    x, y, z = point
    if z <= 0:
        return None
    u = int((fx * x) / z + cx)
    v = int((fy * y) / z + cy)
    return u, v

# ---------- 5. Overlay Points on Camera Image ----------
for point in obstacle_points:
    projection = project_point(point)
    if projection:
        u, v = projection
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # Draw red circles for obstacles
            cv2.circle(image, (u, v), 3, (0, 0, 255), -1)

# ---------- 6. Display the Result ----------
cv2.imshow("Fused Obstacle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
