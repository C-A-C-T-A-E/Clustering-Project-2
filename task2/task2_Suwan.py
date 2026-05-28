import os
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt

# =========================================================================
# STEP 1: Extract grid coordinates inside South Korea territory
# =========================================================================

# Load territory boundary data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path:=os.path.join(base_dir, 'Data_South_Korea_territory.csv'))
df = pd.read_csv(file_path)

lon = df['Longitude (deg)'].values
lat = df['Latitude (deg)'].values

# Generate a polygon object based on the territory boundary data
territory_polygon = shapely.Polygon(list(zip(lon, lat)))

# Generate a regular mesh grid covering the entire bounding box
grid_size = 100 
lon_linspace = np.linspace(lon.min(), lon.max(), grid_size)
lat_linspace = np.linspace(lat.min(), lat.max(), grid_size)
lon_grid, lat_grid = np.meshgrid(lon_linspace, lat_linspace)

# Convert all grid points to Shapely Point objects and identify points inside the territory
points_grid = shapely.points(lon_grid, lat_grid)
mask = shapely.contains(territory_polygon, points_grid)

# Extract coordinates located inside the territory boundary
inside_lon = lon_grid[mask]
inside_lat = lat_grid[mask]

# Convert the filtered inside points into a DataFrame
output_df = pd.DataFrame({
    'Longitude (deg)': inside_lon,
    'Latitude (deg)': inside_lat
})

# Format input data for clustering
points = output_df[['Longitude (deg)', 'Latitude (deg)']].values


# =========================================================================
# STEP 2: Plug the extracted coordinates into K-Means Clustering
# =========================================================================
k = 10
np.random.seed(42)  # Fixed seed for reproducibility
random_indices = np.random.choice(len(points), k, replace=False)
centroids = points[random_indices]

tol = 1e-4
iterations_taken = 0

while True:
    iterations_taken += 1
    clusters = []

    # Assign each point to the nearest centroid
    for point in points:
        distances = np.linalg.norm(point - centroids, axis=1)
        cluster = np.argmin(distances)
        clusters.append(cluster)

    clusters = np.array(clusters)

    # Calculate new centroids
    new_centroids = []
    for i in range(k):
        cluster_points = points[clusters == i]
        if len(cluster_points) == 0:
            new_centroid = centroids[i]
        else:
            new_centroid = cluster_points.mean(axis=0)
        new_centroids.append(new_centroid)

    new_centroids = np.array(new_centroids)

    # Check for convergence (if centroids stop moving)
    if np.allclose(centroids, new_centroids, atol=tol):
        centroids = new_centroids
        break

    centroids = new_centroids

print(f"⚡ K-Means Clustering complete! Converged after {iterations_taken} iterations.")


# =========================================================================
# STEP 3: 4-Panel Visualization (Data preprocessing flow from Left to Right)
# =========================================================================
# Arrange 4 subplots in 1 row (Width increased to 28 for clarity)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 8))

# -------------------------------------------------------------------------
# [Plot 1] Raw Territory Boundary Data
# -------------------------------------------------------------------------
ax1.plot(lon, lat, color='darkred', linewidth=2, label='Territory Boundary')
ax1.fill(lon, lat, color='pink', alpha=0.3, label='Territory Area')
ax1.set_title('1. Raw Territory Boundary', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Latitude (deg)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------------
# [Plot 2] Full Mesh Grid over the bounding box
# -------------------------------------------------------------------------
ax2.scatter(lon_grid, lat_grid, color='gray', s=4, alpha=0.3, label='Generated Grid')
ax2.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax2.set_title('2. Full Mesh Grid (100x100)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude (deg)')
ax2.set_ylabel('Latitude (deg)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------------
# [Plot 3] Grid points filtered inside the territory boundary
# -------------------------------------------------------------------------
ax3.scatter(
    points[:, 0], points[:, 1], 
    color='navy', s=10, alpha=0.4, 
    label='Filtered Grid Points'
)
ax3.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax3.set_title('3. Filtered Grid (Inside Territory Only)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude (deg)')
ax3.set_ylabel('Latitude (deg)')
ax3.legend(loc='upper left')
ax3.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------------
# [Plot 4] Final K-Means Clustering Result
# -------------------------------------------------------------------------
cmap = plt.colormaps['tab10'].resampled(k)

# Plot 10 clusters
for i in range(k):
    cluster_points = points[clusters == i]
    ax4.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        color=cmap(i), s=10, alpha=0.6,
        label=f'Cluster {i+1}'
    )

# Plot final 10 centroids (Star markers)
ax4.scatter(
    centroids[:, 0], centroids[:, 1],
    color='red', marker='*', s=200, edgecolors='black', zorder=5,
    label='Centroids'
)
ax4.set_title('4. Final K-Means Clustering (K=10)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Longitude (deg)')
ax4.set_ylabel('Latitude (deg)')
# Position legend outside the plot for better visibility
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
ax4.grid(True, linestyle='--', alpha=0.5)

# Final layout adjustment and display
plt.tight_layout()
plt.show()
