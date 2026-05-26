import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans


def sample_points_in_polygon(polygon: Polygon, n_points: int, seed: int = 42):
    """
    Rejection-sample approximately uniform points inside a polygon by sampling
    in the polygon's bounding box and keeping points that fall inside.
    """
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = polygon.bounds
    points = []

    # Oversample in batches to reduce Python overhead
    batch = max(5000, n_points)
    while len(points) < n_points:
        xs = rng.uniform(minx, maxx, size=batch)
        ys = rng.uniform(miny, maxy, size=batch)
        for x, y in zip(xs, ys):
            if polygon.contains(Point(x, y)):
                points.append((x, y))
                if len(points) >= n_points:
                    break
    return np.array(points)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evenly distributed points inside the South Korea territory polygon."
    )
    parser.add_argument(
        "territory_csv",
        help="Path to the territory boundary CSV file.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10,
        help="Number of output centroid points to generate.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent

    # ---- 1) Load territory boundary ----
    territory = pd.read_csv(Path(args.territory_csv))
    # expected columns: Longitude (deg), Latitude (deg)
    boundary = territory[["Longitude (deg)", "Latitude (deg)"]].to_numpy()

    # Build polygon from boundary points
    poly = Polygon(boundary)
    if not poly.is_valid:
        # Try to fix self-intersections etc.
        poly = poly.buffer(0)

    # ---- 2) Create many interior sample points ----
    # Bigger number => smoother/more stable centroid placement, but slower
    n_interior = 20000
    interior = sample_points_in_polygon(poly, n_interior, seed=42)

    # ---- 3) KMeans to evenly distribute N points ----
    N = args.n_points
    km = KMeans(n_clusters=N, random_state=42, n_init="auto")
    km.fit(interior)
    centroids = km.cluster_centers_

    # ---- 4) Output results ----
    centroids_df = pd.DataFrame(centroids, columns=["Longitude (deg)", "Latitude (deg)"])
    output_path = base_dir / f"task2_centroids_N{N}.csv"
    centroids_df.to_csv(output_path, index=False)

    print("Centroids (final evenly distributed sample points):")
    print(centroids_df)

    # ---- 5) Plot ----
    plt.figure(figsize=(8, 10))
    # territory outline
    plt.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1, label="Territory boundary")
    # interior sampled points (light)
    plt.scatter(interior[:, 0], interior[:, 1], s=1, alpha=0.05, label="Interior samples")
    # centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=120, color="red", edgecolor="white", linewidth=1.5, label="KMeans centroids")

    plt.title(f"Task 2: Even distribution of N={N} points (KMeans on uniform interior samples)")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    output_png = base_dir / f"task2_centroids_N{N}.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot: {output_png}")
    plt.show()


if __name__ == "__main__":
    main()