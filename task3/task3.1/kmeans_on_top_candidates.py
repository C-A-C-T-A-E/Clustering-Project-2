import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine_km(lon1, lat1, lon2, lat2):
    # all args numpy arrays (or scalars); returns distance in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c

def nearest_indices(centroids, candidates_coords):
    # centroids: (k,2), candidates_coords: (n,2)
    idxs = []
    cand_lon = candidates_coords[:,0]
    cand_lat = candidates_coords[:,1]
    for lon_c, lat_c in centroids:
        d = haversine_km(lon_c, lat_c, cand_lon, cand_lat)
        idxs.append(int(np.argmin(d)))
    return np.array(idxs, dtype=int)

def main():
    parser = argparse.ArgumentParser(description="KMeans on Top candidates and snap centroids to nearest real candidate")
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    default_candidates = script_dir / "produced_data" / "Top_400_Candidates.csv"
    default_territory = project_dir / "data" / "Data_South_Korea_territory.csv"

    parser.add_argument("--candidates", default=str(default_candidates), help="Path to candidates CSV")
    parser.add_argument("--k", type=int, default=17, help="Number of clusters / desired vertiports")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--out-dir", default=None, help="Directory to write outputs (default: task3.1/produced_data)")
    parser.add_argument("--territory", default=str(default_territory), help="Path to territory polygon CSV")
    parser.add_argument("--no-show", action="store_true", help="Do not open the plot window interactively")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "produced_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    # resolve candidates path robustly
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        alt_data = project_dir / "data" / candidates_path.name
        alt_local = script_dir / candidates_path.name
        if alt_data.exists():
            candidates_path = alt_data
        elif alt_local.exists():
            candidates_path = alt_local
        else:
            raise FileNotFoundError(
                f"Candidates CSV not found: {args.candidates}. Tried {alt_data} and {alt_local}."
            )

    # read candidates (coerce lon/lat to numeric if necessary)
    df = pd.read_csv(candidates_path)
    # detect longitude/latitude columns (allow variants like 'Longitude (deg)')
    cols = list(df.columns)
    lon_col = next((c for c in cols if "long" in c.lower()), None)
    lat_col = next((c for c in cols if "lat" in c.lower()), None)
    if lon_col is None or lat_col is None:
        raise SystemExit("candidates CSV must contain longitude and latitude columns")

    coords = df[[lon_col, lat_col]].to_numpy(dtype=float)

    # load territory polygon (resolve path relative to project if needed)
    territory_coords = None
    terr_path = Path(args.territory)
    if not terr_path.exists():
        # try relative to project dir
        terr_path = project_dir / "data" / terr_path.name
    if terr_path.exists():
        terr_df = pd.read_csv(terr_path)
        terr_cols = list(terr_df.columns)
        terr_lon = next((c for c in terr_cols if "long" in c.lower()), None)
        terr_lat = next((c for c in terr_cols if "lat" in c.lower()), None)
        if terr_lon and terr_lat:
            # coerce to numeric and drop NaNs
            terr_df[terr_lon] = pd.to_numeric(terr_df[terr_lon], errors="coerce")
            terr_df[terr_lat] = pd.to_numeric(terr_df[terr_lat], errors="coerce")
            terr_df = terr_df.dropna(subset=[terr_lon, terr_lat])
            territory_coords = terr_df[[terr_lon, terr_lat]].to_numpy(dtype=float)
            # ensure polygon is closed for plotting
            if len(territory_coords) > 0 and not np.allclose(territory_coords[0], territory_coords[-1]):
                territory_coords = np.vstack([territory_coords, territory_coords[0]])

    try:
        from sklearn.cluster import KMeans
    except Exception:
        raise SystemExit("scikit-learn is required to run this script. Install via 'pip install scikit-learn'.")

    kmeans = KMeans(n_clusters=args.k, random_state=args.random_state, n_init=args.n_init)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_

    idxs = nearest_indices(centroids, coords)

    centroids_df = pd.DataFrame(centroids, columns=["Longitude", "Latitude"]) 
    centroids_path = out_dir / f"topcandidate_kmeans_centroids_k{args.k}.csv"
    centroids_df.to_csv(centroids_path, index=False)

    selected = df.iloc[idxs].reset_index(drop=True)
    selected_path = out_dir / "topcandidate_final_vertiport_sites.csv"
    selected.to_csv(selected_path, index=False)

    print(f"Saved centroids: {centroids_path}")
    print(f"Saved selected sites: {selected_path}")

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot territory polygon first so the rest appears on top
    if territory_coords is not None:
        ax.plot(
            territory_coords[:, 0],
            territory_coords[:, 1],
            color='green',
            linewidth=1.0,
            label='Territory outline',
            zorder=0,
        )
        # Use combined bounds so out-of-territory candidates (for example Jeju)
        # are still visible even if the territory polygon itself does not cover them.
        all_lon = np.concatenate([territory_coords[:, 0], coords[:, 0]])
        all_lat = np.concatenate([territory_coords[:, 1], coords[:, 1]])
        lon_min, lon_max = np.min(all_lon), np.max(all_lon)
        lat_min, lat_max = np.min(all_lat), np.max(all_lat)
        pad_lon = max(0.15, (lon_max - lon_min) * 0.03)
        pad_lat = max(0.15, (lat_max - lat_min) * 0.03)
        ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

    # plot candidates (light grey points)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c='lightgrey', alpha=0.55, label='candidates', zorder=1)
    # plot centroids (red X)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=140, c='red', marker='x', linewidths=2.0, label='centroids', zorder=3)
    # plot selected sites (green circles)
    sel_coords = selected[[lon_col, lat_col]].to_numpy(dtype=float)
    ax.scatter(sel_coords[:, 0], sel_coords[:, 1], s=140, c='blue', marker='*', edgecolor='black', linewidth=0.6, label='selected sites', zorder=4)

    # annotate centroid indices slightly offset for readability
    for i, (cx, cy) in enumerate(centroids):
        ax.annotate(str(i), (cx + 0.04, cy + 0.04), color='red', fontsize=9, zorder=5)

    ax.set_title(f'KMeans (k={args.k}) on Top Candidates')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    output_png = out_dir / "topcandidate_kmeans_plot.png"
    fig.savefig(output_png, dpi=150)
    print(f"Saved figure: {output_png}")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
