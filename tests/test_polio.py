from laser_polio import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
