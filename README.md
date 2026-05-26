# Clustering Project 2

This repository contains our team’s full submission for the vertiport clustering project. We organized the work into three tasks and documented the methodology, intermediate outputs, and final comparison results.

## Our Approach

We solved the project in a staged way rather than jumping directly to the final vertiport sites.

1. **Task 1**: we first implemented K-means manually on a small toy dataset to demonstrate that we understood the algorithm itself.
2. **Task 2**: we then used the South Korea territory boundary to generate evenly distributed points inside the polygon so that we could simulate a practical spatial placement strategy.
3. **Task 3**: finally, we built a scoring pipeline for candidate vertiports using traffic, hospital, and tourist attraction data. We ranked all candidates, kept the top candidates, and compared two final-selection approaches:
   - K-means on the full candidate list, and
   - K-means only on the top-ranked candidate list.

We prefer the top-candidate approach as the final recommendation because it combines multiple datasets before clustering, which makes the result more efficient and easier to justify.

## Task Documents

The detailed explanation for each task is written in the task-specific README files:

- [Task 1](task1/README.md)
- [Task 2](task2/README.md)
- [Task 3](task3/README.md)

## Repository Layout

```text
.
├─ data/                         raw input data and generated score tables
├─ task1/                        toy K-means implementation
├─ task2/                        territory-based even distribution script
├─ task3/
│  ├─ task3.1/                   scoring and vertiport selection scripts
│  └─ task3.2/                   reserved for future work
└─ Project_2_description.pdf     assignment description
```

## Main Outputs

- `task1/task1_figure.webp`
- `task2/task2_centroids_N*.csv`
- `task2/task2_centroids_N*.png`
- `task3/task3.1/produced_data/Processed_Data_vertiport_candidates_scores.csv`
- `task3/task3.1/produced_data/Top_400_Candidates.csv`
- `task3/task3.1/produced_data/allcandidate_final_vertiport_sites.csv`
- `task3/task3.1/produced_data/topcandidate_final_vertiport_sites.csv`
- `task3/task3.1/produced_data/allcandidate_kmeans_plot.png`
- `task3/task3.1/produced_data/topcandidate_kmeans_plot.png`

## Environment

Recommended packages:

- pandas
- numpy
- matplotlib
- scikit-learn
- shapely

Install:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib scikit-learn shapely
```

## Quick Run

```powershell
# Task 1
py -3 task1\task1.py

# Task 2
py -3 task2\even_distribution.py data\Data_South_Korea_territory.csv --n-points 17

# Task 3 scoring
py -3 task3\task3.1\build_candidate_scores.py --candidates-csv data\Data_vertiport_candidates.csv --hospitals-csv data\General_Hospitals_Coordinates.csv --tour-csv data\Tourist_Attraction_Data.csv --traffic-csv data\Traffic_Data.csv --output-processed task3\task3.1\produced_data\Processed_Data_vertiport_candidates_scores.csv --top-n 400 --output-top task3\task3.1\produced_data\Top_400_Candidates.csv

# Task 3 full candidate comparison
py -3 task3\task3.1\kmeans_17_vertiports.py data\Data_vertiport_candidates.csv data\Data_South_Korea_territory.csv --k 17 --random-state 42

# Task 3 top candidate comparison
py -3 task3\task3.1\kmeans_on_top_candidates.py --candidates task3\task3.1\produced_data\Top_400_Candidates.csv --k 17 --random-state 42
```

## Notes

- `Top_400_Candidates.csv` is created by scoring all candidates and sorting by `Total_Score`.
- Some Jeju candidates appear outside the plotted polygon because the provided territory boundary file is not a perfect Jeju boundary.
- `task3/task3.1/produced_data/` contains the comparison-ready outputs for both approaches.
