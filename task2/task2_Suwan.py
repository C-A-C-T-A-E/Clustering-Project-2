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
file_path = os.path.join(base_dir, 'Data_South_Korea_territory.csv')
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
# STEP 2: K-Means Clustering Function
# =========================================================================
def run_kmeans(points, k, tol=1e-4, seed=42):
    np.random.seed(seed)  # Fixed seed for reproducibility
    random_indices = np.random.choice(len(points), k, replace=False)
    centroids = points[random_indices]
    
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
        
    print(f"⚡ K-Means (K={k}) complete! Converged after {iterations_taken} iterations.")
    return clusters, centroids


# =========================================================================
# STEP 3: FIGURE 1 - Data Preprocessing Steps (3 Panels)
# =========================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
fig1.suptitle('Data Preprocessing Flow', fontsize=16, fontweight='bold', y=0.98)

# [Plot 1] Raw Territory Boundary Data
ax1.plot(lon, lat, color='darkred', linewidth=2, label='Territory Boundary')
ax1.fill(lon, lat, color='pink', alpha=0.3, label='Territory Area')
ax1.set_title('1. Raw Territory Boundary', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Latitude (deg)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# [Plot 2] Full Mesh Grid over the bounding box
ax2.scatter(lon_grid, lat_grid, color='gray', s=4, alpha=0.3, label='Generated Grid')
ax2.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax2.set_title('2. Full Mesh Grid (100x100)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude (deg)')
ax2.set_ylabel('Latitude (deg)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# [Plot 3] Grid points filtered inside the territory boundary
ax3.scatter(points[:, 0], points[:, 1], color='navy', s=10, alpha=0.4, label='Filtered Grid Points')
ax3.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax3.set_title('3. Filtered Grid (Inside Only)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude (deg)')
ax3.set_ylabel('Latitude (deg)')
ax3.legend(loc='upper left')
ax3.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()


# =========================================================================
# STEP 4: FIGURE 2 - Final K-Means Clustering Comparison (3 Panels)
# =========================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 7))
fig2.suptitle('K-Means Clustering Comparison (N = 3, 5, 10)', fontsize=16, fontweight='bold', y=0.98)

k_values = [3, 5, 10]

for idx, k in enumerate(k_values):
    ax = axes2[idx]
    
    # Run clustering algorithm
    clusters, centroids = run_kmeans(points, k=k)
    cmap = plt.colormaps['tab10'].resampled(k)
    
    # Plot data points for each cluster
    for i in range(k):
        cluster_points = points[clusters == i]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],  # x: Longitude(0), y: Latitude(1)
            color=cmap(i), s=10, alpha=0.6,
            label=f'C {i+1}'
        )
        
    # Plot final centroids (Star markers)
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        color='red', marker='*', s=150, edgecolors='black', zorder=5,
        label='Centroids'
    )
    
    ax.set_title(f'K = {k} Clustering Result', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    
    # Split legend into 2 columns if K > 5 to prevent overlap with spatial data
    ncol = 2 if k > 5 else 1
    ax.legend(loc='upper left', fontsize=9, ncol=ncol)
    ax.grid(True, linestyle='--', alpha=0.5)

fig2.tight_layout()

# Display both figure windows on the screen
plt.show()
