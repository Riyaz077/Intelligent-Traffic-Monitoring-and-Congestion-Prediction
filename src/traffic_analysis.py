"""Traffic count aggregation and configurable congestion heuristic."""
import pandas as pd

def classify_congestion(vehicle_count, low_threshold=5, high_threshold=12):
    """Classify sampled-frame vehicle count; thresholds need site calibration."""
    if vehicle_count < low_threshold:
        return "Low"
    if vehicle_count < high_threshold:
        return "Moderate"
    return "High"

def build_record(frame_number, timestamp_seconds, counts, low_threshold=5, high_threshold=12):
    total = int(sum(counts.values()))
    return {
        "frame": int(frame_number),
        "timestamp_seconds": round(float(timestamp_seconds), 2),
        **{f"{k}_count": int(v) for k, v in counts.items()},
        "total_vehicles": total,
        "congestion_level": classify_congestion(total, low_threshold, high_threshold),
    }

def summarize_results(records):
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
