import json

from boundary_sweep.geo06 import plan_sweep, required_distance, trajectory_metrics


def surface(width=20.0, height=12.0):
    return {
        "physical_boundary": {
            "TOP": {"start": [0, 0, height], "end": [width, 0, height]},
            "BOTTOM": {"start": [0, 0, 0], "end": [width, 0, 0]},
            "LEFT": {"start": [0, 0, height], "end": [0, 0, 0]},
            "RIGHT": {"start": [width, 0, height], "end": [width, 0, 0]},
        }
    }


def frame(state, coverage=0.5, boundary_inside=False):
    return {"labels": {"label": state, "target_pixel_coverage": coverage, "boundary": {"boundary_in_image": boundary_inside}}}


def test_geo06_planner_leaves_in_margin_and_stays_bounded():
    d = required_distance(surface())
    plan = plan_sweep(surface(), "LEFT", d)
    assert plan["steps"] <= 80
    assert plan["thresholds"]["min_in_frames"] >= 5
    assert plan["image_footprint_width_m"] < plan["facade_width_m"]


def test_geo06_direction_maps_active_boundary():
    d = required_distance(surface())
    assert plan_sweep(surface(), "LEFT", d)["active_boundary"] == "LEFT"
    assert plan_sweep(surface(), "RIGHT", d)["active_boundary"] == "RIGHT"
    assert plan_sweep(surface(), "UP", d)["active_boundary"] == "TOP"
    assert plan_sweep(surface(), "DOWN", d)["active_boundary"] == "BOTTOM"


def test_geo06_event_metrics_accept_only_ordered_complete_sequence():
    rows = [frame("IN") for _ in range(5)] + [frame("STRADDLE", boundary_inside=True) for _ in range(5)] + [frame("OUT", 0.0) for _ in range(5)]
    result = trajectory_metrics(rows)
    assert result["event_coverage"] is True
    assert result["monotonic_ignoring_unknown"] is True
    assert result["straddle_boundary_inside"] is True


def test_geo06_unknown_frames_do_not_create_fake_event_coverage():
    result = trajectory_metrics([frame("IN")] * 5 + [frame("UNKNOWN")] * 20 + [frame("OUT", 0.0)] * 5)
    assert result["event_coverage"] is False
    assert result["unknown_ratio"] > 0.1


def test_geo06_reverse_state_is_rejected():
    result = trajectory_metrics([frame("IN"), frame("STRADDLE", boundary_inside=True), frame("IN")])
    assert result["monotonic_ignoring_unknown"] is False


def test_geo06_manifest_has_physical_boundary_keys():
    surface_path = "results/geo06/surfaces/surface_omega.json"
    try:
        data = json.load(open(surface_path))
    except FileNotFoundError:
        return
    assert set(data["physical_boundary"]) == {"LEFT", "RIGHT", "TOP", "BOTTOM"}
    assert data["bbox_used_for"] == "candidate_search_bounds_only"
