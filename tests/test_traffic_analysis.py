from src.traffic_analysis import classify_congestion, build_record, summarize_results

def test_congestion_thresholds():
    assert classify_congestion(2, 5, 12) == "Low"
    assert classify_congestion(7, 5, 12) == "Moderate"
    assert classify_congestion(15, 5, 12) == "High"

def test_record_and_summary():
    row = build_record(15, 0.5, {"car": 2, "bus": 1, "truck": 0, "motorcycle": 1})
    df = summarize_results([row])
    assert df.iloc[0]["total_vehicles"] == 4
    assert len(df) == 1
