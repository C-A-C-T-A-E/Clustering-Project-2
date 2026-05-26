import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0

def find_lon_lat_cols(df):
    cols = [c.lower() for c in df.columns]
    lon_idx = None
    lat_idx = None
    for i,c in enumerate(cols):
        if 'lon' in c or 'longitude' in c:
            lon_idx = i
        if 'lat' in c or 'latitude' in c:
            lat_idx = i
    if lon_idx is None or lat_idx is None:
        raise ValueError('Could not find lon/lat columns')
    return df.columns[lon_idx], df.columns[lat_idx]

def to_radians(coords):
    return np.radians(coords)

def average_candidate_distance_km(coords_deg, centroids_deg):
    # coords_deg: Nx2 (lon,lat)
    # convert to radians and use BallTree haversine
    coords_rad = np.radians(coords_deg[:,[1,0]])  # lat, lon order for haversine
    cents_rad = np.radians(centroids_deg[:,[1,0]])
    tree = BallTree(coords_rad, metric='haversine')
    dists, idx = tree.query(cents_rad, k=coords_rad.shape[0])
    # For coverage metric, compute for each candidate the distance to nearest centroid
    # Use a tree of centroids instead
    cent_tree = BallTree(cents_rad, metric='haversine')
    dists_c, _ = cent_tree.query(coords_rad, k=1)
    d_km = dists_c[:,0] * EARTH_KM
    return float(d_km.mean())

def run_k_sweep(candidates_csv, k_min, k_max, step, output_dir, random_state=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(candidates_csv)
    lon_col, lat_col = find_lon_lat_cols(df)
    coords = df[[lon_col, lat_col]].to_numpy(dtype=float)

    ks = list(range(k_min, k_max+1, step))
    records = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(coords)
        inertia = float(km.inertia_)
        sil = None
        try:
            if k > 1 and len(coords) > k:
                sil = float(silhouette_score(coords, labels))
        except Exception:
            sil = None
        avg_dist_km = average_candidate_distance_km(coords, km.cluster_centers_)
        records.append({'k': k, 'inertia': inertia, 'silhouette': sil, 'avg_candidate_dist_km': avg_dist_km})
    out_df = pd.DataFrame.from_records(records)
    csv_out = output_dir / f'k_sweep_{Path(candidates_csv).stem}.csv'
    png_out = output_dir / f'k_sweep_{Path(candidates_csv).stem}.png'
    out_df.to_csv(csv_out, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(out_df['k'], out_df['inertia'], '-o', label='inertia')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia (sum squared dist)')
    ax2 = ax1.twinx()
    ax2.plot(out_df['k'], out_df['avg_candidate_dist_km'], '-s', color='C1', label='avg dist (km)')
    ax2.set_ylabel('Avg candidate -> centroid dist (km)')
    plt.title(f'k-sweep: {Path(candidates_csv).name}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='best')
    plt.tight_layout()
    fig.savefig(png_out)
    plt.close(fig)
    return csv_out, png_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', default='task3/task3.1/produced_data/Top_400_Candidates.csv')
    p.add_argument('--all-candidates', default='data/Data_vertiport_candidates.csv')
    p.add_argument('--k-min', type=int, default=5)
    p.add_argument('--k-max', type=int, default=30)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--output-dir', default='task3/task3.2/produced_data')
    p.add_argument('--random-state', type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    # run on top candidates
    print('Running k-sweep on top candidates:', args.candidates)
    csv_top, png_top = run_k_sweep(args.candidates, args.k_min, args.k_max, args.step, out_dir, args.random_state)
    print('Saved:', csv_top, png_top)

    # run on all candidates
    print('Running k-sweep on all candidates:', args.all_candidates)
    csv_all, png_all = run_k_sweep(args.all_candidates, args.k_min, args.k_max, args.step, out_dir, args.random_state)
    print('Saved:', csv_all, png_all)

if __name__ == '__main__':
    main()
