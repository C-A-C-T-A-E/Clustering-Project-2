# Task 3.2: Choosing k (k-sweep)

This folder contains the k-sweep utility used to help choose the number of clusters (`k`) for Task 3.

Script:

- `task3/task3.2/find_optimal_k.py`

Purpose:

- Sweep a range of `k` values and record clustering metrics:
  - `inertia` (sum of squared distances)
  - `silhouette` (when computable)
  - `avg_candidate_dist_km` (average distance from candidates to nearest centroid, in km)

Usage (example):

```powershell
py -3 task3\task3.2\find_optimal_k.py --k-min 5 --k-max 30 --step 1 --output-dir task3\task3.2\produced_data
```

Outputs (saved to `task3/task3.2/produced_data` by default):

- `k_sweep_Top_400_Candidates.csv` / `.png` — results for the Top-N shortlist
- `k_sweep_Data_vertiport_candidates.csv` / `.png` — results for the full candidate set

Notes:

- The script auto-detects longitude/latitude columns from the CSV header.
- The produced PNG shows inertia and average candidate→centroid distance to help identify an elbow or coverage trade-off.
