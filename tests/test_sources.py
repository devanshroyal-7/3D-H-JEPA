from point_hjepa.data.planners import create_planner
from point_hjepa.data.sources import assign_sources, source_counts
from point_hjepa.data.tasks import TaskUnavailable, require_task


def test_mixture_is_exact_for_twenty_episodes():
    sources = assign_sources(20, {"expert": 0.5, "noisy": 0.3, "explore": 0.2}, seed=0)
    assert source_counts(sources) == {"expert": 10, "noisy": 6, "explore": 4}
    assert len(sources) == 20


def test_default_planner_is_srmp_and_mplib_is_optional():
    assert create_planner("srmp").name == "srmp"
    assert create_planner("mplib").name == "mplib"


def test_settable_is_stubbed_with_a_reason():
    try:
        require_task("SetTable-v1")
    except TaskUnavailable as exc:
        message = str(exc)
    else:
        raise AssertionError("SetTable should not be collected")
    assert "ManiSkill-HAB" in message
    assert "mplib" not in message.lower() or "not" in message.lower()
