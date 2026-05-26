import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from sklearn.cluster import KMeans

def main():
    parser = argparse.ArgumentParser(
        description="Cluster vertiport candidates and pick the nearest real candidate to each centroid."
    )
    parser.add_argument(
        "candidates_csv",
        help="Path to the vertiport candidates CSV file.",
    )
    parser.add_argument(
        "territory_csv",
        help="Path to the South Korea territory boundary CSV file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=17,
        help="Number of clusters / vertiport locations to generate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by KMeans.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where output CSV/PNG files are written (default: task3.1/produced_data).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the plot window interactively.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    candidates_path = Path(args.candidates_csv)
    territory_path = Path(args.territory_csv)
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "produced_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_path.exists():
        alt = project_dir / "data" / candidates_path.name
        if alt.exists():
            candidates_path = alt
    if not territory_path.exists():
        alt = project_dir / "data" / territory_path.name
        if alt.exists():
            territory_path = alt

    candidates = pd.read_csv(candidates_path)
    territory = pd.read_csv(territory_path)

    coords = candidates[["Longitude (deg)", "Latitude (deg)"]].to_numpy()

    kmeans = KMeans(n_clusters=args.k, random_state=args.random_state, n_init="auto")
    labels = kmeans.fit_predict(coords)

    centroids = kmeans.cluster_centers_  # shape (K, 2): [lon, lat]

    centroids_df = pd.DataFrame(centroids, columns=["Longitude (deg)", "Latitude (deg)"])
    centroids_df.index.name = "cluster_id"
    output_centroids_csv = out_dir / f"allcandidate_kmeans_centroids_k{args.k}.csv"
    centroids_df.to_csv(output_centroids_csv)
    print(f"Saved KMeans centroids to: {output_centroids_csv}")

    final_sites = []
    for cluster_id in range(args.k):
        cluster_mask = labels == cluster_id
        cluster_points = coords[cluster_mask]

        diff = cluster_points - centroids[cluster_id]
        d2 = np.sum(diff * diff, axis=1)
        local_idx = np.argmin(d2)
        chosen_point = cluster_points[local_idx]

        final_sites.append({
            "cluster_id": cluster_id,
            "centroid_lon": centroids[cluster_id, 0],
            "centroid_lat": centroids[cluster_id, 1],
            "final_site_lon": chosen_point[0],
            "final_site_lat": chosen_point[1],
            "cluster_size": int(cluster_points.shape[0]),
            "centroid_to_site_dist_deg": float(np.sqrt(d2[local_idx])),
        })

    final_sites_df = pd.DataFrame(final_sites).sort_values("cluster_id")
    output_final_sites_csv = out_dir / "allcandidate_final_vertiport_sites.csv"
    final_sites_df.to_csv(output_final_sites_csv, index=False)

    print(f"Saved final vertiport sites (chosen from candidates) to: {output_final_sites_csv}")
    print(final_sites_df[["cluster_id", "final_site_lon", "final_site_lat", "cluster_size", "centroid_to_site_dist_deg"]])

    plt.figure(figsize=(8, 10))
    plt.plot(
        territory["Longitude (deg)"],
        territory["Latitude (deg)"],
        color="black",
        linewidth=1.0,
        label="Territory outline",
    )
    plt.scatter(
        candidates["Longitude (deg)"],
        candidates["Latitude (deg)"],
        s=10,
        c=labels,
        cmap="tab20",
        alpha=0.55,
        label="Candidates (colored by cluster)",
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=160,
        c="red",
        marker="X",
        edgecolor="white",
        linewidth=1.0,
        label=f"KMeans centroids (K={args.k})",
    )
    plt.scatter(
        final_sites_df["final_site_lon"],
        final_sites_df["final_site_lat"],
        s=140,
        c="limegreen",
        marker="o",
        edgecolor="black",
        linewidth=1.0,
        label="Final sites (nearest candidate to centroid)",
    )

    plt.title(f"Task 3 (Part 1): KMeans (K={args.k}) + nearest-candidate final sites")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_png = out_dir / "allcandidate_kmeans_plot.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved figure to: {output_png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()