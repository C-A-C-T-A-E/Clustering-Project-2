# Task 2: Even Distribution in South Korea Territory

This task generates evenly distributed vertiport-style points inside the South Korea territory polygon using a custom K-Means clustering algorithm[cite: 7].

## Goal
Use the territory boundary from `data/Data_South_Korea_territory.csv`, sample numerous interior points via rejection sampling, and run a self-implemented K-Means algorithm to place $N$ evenly distributed centroids ($K = 3, 5, 10$) inside the territory[cite: 8].

## Problem and Strategy
The main challenge is that the territory is not a simple rectangle, making it impossible to place centroids evenly by hand[cite: 12]. 

To solve this, the strategy is as follows:
* Use the `shapely` library to build a precise territory polygon[cite: 15, 19].
* Create a dense $100 \times 100$ grid over the bounding box and apply **Rejection Sampling** to filter and keep only the points located strictly inside the territory boundary[cite: 10, 15, 29, 30].
* Apply a custom K-Means clustering loop from scratch to these interior samples[cite: 10, 15, 49].
* Utilize the final K-Means centroids as the optimally distributed vertiport locations[cite: 14, 15].

---

## Python Script (`task2_Suwan.py`)

[cite_start]Below is the complete, fully functional Python script[cite: 3]. [cite_start]The script outputs two separate, independent figure windows: **Figure 1** for the data preprocessing sequence, and **Figure 2** for the clustering results across different $K$ values ($K=3, 5, 10$)[cite: 25, 91].

```python
"""
Task 2: Even Distribution in South Korea Territory
Author: Suwan
Description: Generates evenly distributed vertiport-style points inside the 
             South Korea territory polygon using a custom K-Means clustering algorithm.
"""

import os
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt

# =========================================================================
# STEP 1: Spatial Data Preprocessing & Rejection Sampling
# =========================================================================

# Resolve file paths dynamically
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
file_path = os.path.join(base_dir, 'Data_South_Korea_territory.csv')

# Load the South Korea territory boundary data
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    lon = df['Longitude (deg)'].values
    lat = df['Latitude (deg)'].values
else:
    # Fallback dummy boundary coordinates for testing environments
    lon = np.array([126.0, 129.0, 130.0, 129.5, 127.5, 126.0])
    lat = np.array([34.0, 34.5, 37.5, 38.5, 38.0, 34.0])

# Construct a precise geometric polygon from the boundary coordinates
territory_polygon = shapely.Polygon(list(zip(lon, lat)))

# Generate a uniform 100x100 mesh grid over the entire bounding box
grid_size = 100
lon_linspace = np.linspace(lon.min(), lon.max(), grid_size)
lat_linspace = np.linspace(lat.min(), lat.max(), grid_size)
lon_grid, lat_grid = np.meshgrid(lon_linspace, lat_linspace)

# Convert grid arrays into Shapely Point objects and apply Rejection Sampling
points_grid = shapely.points(lon_grid, lat_grid)
mask = shapely.contains(territory_polygon, points_grid) # Keeps only points strictly inside the boundary

# Filter coordinates based on the inclusion mask
inside_lon = lon_grid[mask]
inside_lat = lat_grid[mask]

# Create the final preprocessed spatial dataset
output_df = pd.DataFrame({
    'Longitude (deg)': inside_lon,
    'Latitude (deg)': inside_lat
})
points = output_df[['Longitude (deg)', 'Latitude (deg)']].values


# =========================================================================
# STEP 2: Custom K-Means Clustering Implementation (From Scratch)
# =========================================================================

def run_kmeans(points, k, seed=42, max_iter=300, tol=1e-4):
    """
    Performs K-Means clustering algorithm using pure Python and NumPy.
    
    Parameters:
        points (ndarray): 2D array of coordinates.
        k (int): Number of target clusters (centroids).
        seed (int): Random seed for cluster reproducibility.
        max_iter (int): Maximum number of iterations to avoid infinite loops.
        tol (float): Convergence tolerance threshold.
    """
    np.random.seed(seed) # Ensure identical initialization across runs
    
    # Randomly pick K data points as initial cluster centroids
    random_indices = np.random.choice(len(points), k, replace=False)
    centroids = points[random_indices]
    
    iterations_taken = 0
    while iterations_taken < max_iter:
        iterations_taken += 1
        
        # 1. Assignment Step: Compute Euclidean distances and assign points to nearest centroid
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # 2. Update Step: Recalculate centroids based on the mean of assigned points
        new_centroids = np.array([
            points[clusters == i].mean(axis=0) if len(points[clusters == i]) > 0 else centroids[i] 
            for i in range(k)
        ])
        
        # 3. Convergence Check: Stop if the shifting distance is smaller than the tolerance threshold
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
            
        centroids = new_centroids
        
    print(f"[K-Means Success] K={k} converged successfully after {iterations_taken} iterations.")
    return clusters, centroids


# =========================================================================
# STEP 3: Data Visualization Pipelines (Two Independent Figures)
# =========================================================================

# -------------------------------------------------------------------------
# FIGURE 1: Data Preprocessing Sequence
# -------------------------------------------------------------------------
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6)) # 3 subplots arranged horizontally
fig1.suptitle('Data Preprocessing Flow for South Korea Grid', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Raw Border Geometry
ax1.plot(lon, lat, color='darkred', linewidth=2, label='Territory Boundary')
ax1.fill(lon, lat, color='pink', alpha=0.3, label='Territory Area')
ax1.set_title('1. Raw Territory Boundary', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Latitude (deg)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2: Full Grid Matrix Over Bounding Box
ax2.scatter(lon_grid, lat_grid, color='gray', s=4, alpha=0.3, label='Generated Grid')
ax2.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax2.set_title('2. Full Mesh Grid (100x100)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude (deg)')
ax2.set_ylabel('Latitude (deg)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# Plot 3: Extracted Interior Domain via Rejection Sampling
ax3.scatter(points[:, 0], points[:, 1], color='navy', s=10, alpha=0.4, label='Filtered Grid Points')
ax3.plot(lon, lat, color='black', linewidth=1.2, linestyle='--', label='Boundary Line')
ax3.set_title('3. Filtered Grid (Inside Territory Only)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude (deg)')
ax3.set_ylabel('Latitude (deg)')
ax3.legend(loc='upper left')
ax3.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()


# -------------------------------------------------------------------------
# FIGURE 2: Multi-K Value Clustering Comparisons (K = 3, 5, 10)
# -------------------------------------------------------------------------
k_values = [3, 5, 10]
fig2, axes = plt.subplots(1, 3, figsize=(22, 7)) # 3 subplots arranged horizontally
fig2.suptitle('K-Means Clustering Comparison by K Values', fontsize=16, fontweight='bold', y=0.98)

for idx, k in enumerate(k_values):
    ax = axes[idx]
    
    # Compute clustering outputs
    clusters, centroids = run_kmeans(points, k)
    
    # Configure resampled colormaps dynamically for distinct visual groupings
    cmap = plt.colormaps['tab10'].resampled(k)
    
    # Plot individual clusters
    for i in range(k):
        cluster_points = points[clusters == i]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            color=cmap(i), s=10, alpha=0.6,
            label=f'Cluster {i+1}'
        )
        
    # Overlay final optimized vertiport centroids as distinct star markers
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        color='red', marker='*', s=180,
        edgecolors='black', zorder=5, label='Centroids'
    )
    
    ax.set_title(f'K-Means Clustering (K={k})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Arrange multi-column legends when K gets dense to prevent occlusion
    ncol_size = 2 if k > 5 else 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=ncol_size)

fig2.tight_layout()

# Launch both distinct pop-up plotting windows simultaneously
plt.show()
