import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088


def norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("(deg)", "")
        .replace("_", "")
    )


def find_col(df: pd.DataFrame, aliases: list[str]) -> str:
    normalized = {norm_col(c): c for c in df.columns}
    for alias in aliases:
        key = norm_col(alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find column. Tried aliases: {aliases}. Available: {list(df.columns)}")


def minmax_scale(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def to_latlon_radians(lonlat: np.ndarray) -> np.ndarray:
    # BallTree with haversine expects [lat, lon] in radians.
    latlon = np.column_stack((lonlat[:, 1], lonlat[:, 0]))
    return np.radians(latlon)


def nearest_distance_and_index_km(query_lonlat: np.ndarray, ref_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = BallTree(to_latlon_radians(ref_lonlat), metric="haversine")
    distances_rad, indices = tree.query(to_latlon_radians(query_lonlat), k=1)
    distances_km = distances_rad[:, 0] * EARTH_RADIUS_KM
    return distances_km, indices[:, 0]


def parse_traffic_values(series: pd.Series) -> np.ndarray:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace('"', "", regex=False)
    )
    values = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)
    return values.to_numpy(dtype=float)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Build candidate scores using traffic, hospitals, and tourist attractions."
    )
    parser.add_argument(
        "--candidates-csv",
        default=str(project_dir / "data" / "Data_vertiport_candidates.csv"),
        help="Path to candidate points CSV.",
    )
    parser.add_argument(
        "--hospitals-csv",
        default=str(project_dir / "data" / "General_Hospitals_Coordinates.csv"),
        help="Path to hospital coordinates CSV.",
    )
    parser.add_argument(
        "--tour-csv",
        default=str(project_dir / "data" / "Tourist_Attraction_Data.csv"),
        help="Path to tourist attraction coordinates CSV.",
    )
    parser.add_argument(
        "--traffic-csv",
        default=str(project_dir / "data" / "Traffic_Data.csv"),
        help="Path to traffic points CSV.",
    )
    parser.add_argument(
        "--output-processed",
        default=str(script_dir / "produced_data" / "Processed_Data_vertiport_candidates_scores.csv"),
        help="Output CSV path for all scored candidates.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=400,
        help="Number of top candidates to export.",
    )
    parser.add_argument(
        "--output-top",
        default=None,
        help="Output CSV path for top-N candidates. If omitted, writes data/Top_<N>_Candidates.csv.",
    )
    parser.add_argument(
        "--dedupe-candidates",
        action="store_true",
        default=True,
        help="Remove duplicate candidate coordinates before scoring (enabled by default).",
    )
    parser.add_argument(
        "--no-dedupe-candidates",
        dest="dedupe_candidates",
        action="store_false",
        help="Keep duplicate candidate coordinates.",
    )

    parser.add_argument("--w-traffic", type=float, default=0.5, help="Weight for traffic score.")
    parser.add_argument("--w-hospital", type=float, default=0.3, help="Weight for hospital score.")
    parser.add_argument("--w-tour", type=float, default=0.2, help="Weight for tourist score.")

    parser.add_argument(
        "--hospital-decay-km",
        type=float,
        default=15.0,
        help="Distance decay scale (km) for hospital influence.",
    )
    parser.add_argument(
        "--tour-decay-km",
        type=float,
        default=20.0,
        help="Distance decay scale (km) for tourist influence.",
    )
    parser.add_argument(
        "--traffic-decay-km",
        type=float,
        default=12.0,
        help="Distance decay scale (km) for traffic influence.",
    )

    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates_csv)
    hospitals = pd.read_csv(args.hospitals_csv)
    tour = pd.read_csv(args.tour_csv)
    traffic = pd.read_csv(args.traffic_csv)

    cand_lon_col = find_col(candidates, ["Longitude (deg)", "longitude", "lon", "경도"])
    cand_lat_col = find_col(candidates, ["Latitude (deg)", "latitude", "lat", "위도"])

    hosp_lon_col = find_col(hospitals, ["경도", "longitude", "Longitude (deg)", "lon"])
    hosp_lat_col = find_col(hospitals, ["위도", "latitude", "Latitude (deg)", "lat"])

    tour_lon_col = find_col(tour, ["경도", "longitude", "Longitude (deg)", "lon"])
    tour_lat_col = find_col(tour, ["위도", "latitude", "Latitude (deg)", "lat"])

    traf_lon_col = find_col(traffic, ["longitude", "경도", "Longitude (deg)", "lon"])
    traf_lat_col = find_col(traffic, ["latitude", "위도", "Latitude (deg)", "lat"])
    traf_val_col = find_col(traffic, ["traffic", "trafficcount", "교통량"])

    if args.dedupe_candidates:
        candidates = candidates.drop_duplicates(subset=[cand_lon_col, cand_lat_col]).reset_index(drop=True)

    cand_lonlat = candidates[[cand_lon_col, cand_lat_col]].to_numpy(dtype=float)
    hosp_lonlat = hospitals[[hosp_lon_col, hosp_lat_col]].to_numpy(dtype=float)
    tour_lonlat = tour[[tour_lon_col, tour_lat_col]].to_numpy(dtype=float)
    traf_lonlat = traffic[[traf_lon_col, traf_lat_col]].to_numpy(dtype=float)

    traffic_values = parse_traffic_values(traffic[traf_val_col])
    traffic_values_norm = minmax_scale(traffic_values)

    hosp_dist_km, _ = nearest_distance_and_index_km(cand_lonlat, hosp_lonlat)
    tour_dist_km, _ = nearest_distance_and_index_km(cand_lonlat, tour_lonlat)
    traf_dist_km, nearest_traf_idx = nearest_distance_and_index_km(cand_lonlat, traf_lonlat)

    nearest_traffic_strength = traffic_values_norm[nearest_traf_idx]

    hospital_raw = np.exp(-hosp_dist_km / max(args.hospital_decay_km, 1e-6))
    tour_raw = np.exp(-tour_dist_km / max(args.tour_decay_km, 1e-6))
    traffic_raw = nearest_traffic_strength * np.exp(-traf_dist_km / max(args.traffic_decay_km, 1e-6))

    traffic_score = minmax_scale(traffic_raw)
    hospital_score = minmax_scale(hospital_raw)
    tour_score = minmax_scale(tour_raw)

    weight_sum = args.w_traffic + args.w_hospital + args.w_tour
    if np.isclose(weight_sum, 0.0):
        raise ValueError("At least one of the weights must be non-zero.")

    w_t = args.w_traffic / weight_sum
    w_h = args.w_hospital / weight_sum
    w_u = args.w_tour / weight_sum

    total_score = w_t * traffic_score + w_h * hospital_score + w_u * tour_score

    result = pd.DataFrame(
        {
            "Longitude (deg)": cand_lonlat[:, 0],
            "Latitude (deg)": cand_lonlat[:, 1],
            "Traffic_Score": traffic_score,
            "Hospital_Score": hospital_score,
            "Tour_Score": tour_score,
            "Total_Score": total_score,
            "Nearest_Traffic_Distance_km": traf_dist_km,
            "Nearest_Hospital_Distance_km": hosp_dist_km,
            "Nearest_Tour_Distance_km": tour_dist_km,
        }
    )

    result_sorted = result.sort_values("Total_Score", ascending=False).reset_index(drop=True)

    output_processed = Path(args.output_processed)
    output_processed.parent.mkdir(parents=True, exist_ok=True)
    result_sorted.to_csv(output_processed, index=False)

    if args.output_top:
        output_top = Path(args.output_top)
    else:
        output_top = output_processed.parent / f"Top_{args.top_n}_Candidates.csv"
    output_top.parent.mkdir(parents=True, exist_ok=True)

    top_n = max(1, int(args.top_n))
    result_sorted.head(top_n).to_csv(output_top, index=False)

    print(f"Saved scored candidates: {output_processed}")
    print(f"Saved top {top_n} candidates: {output_top}")
    print(f"Scored candidate rows: {len(result_sorted)}")
    print(result_sorted.head(10)[["Longitude (deg)", "Latitude (deg)", "Total_Score"]])


if __name__ == "__main__":
    main()
