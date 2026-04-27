import pytest

from pyaugmecon.config import PyAugmeconConfig


def test_solver_options_only_drop_none_values():
    config = PyAugmeconConfig(
        name="opts_test",
        solver_options={
            "none_value": None,
            "zero_value": 0,
            "float_zero": 0.0,
            "false_value": False,
            "string_value": "value",
        },
    )

    assert config.solver_options == {
        "zero_value": 0,
        "float_zero": 0.0,
        "false_value": False,
        "string_value": "value",
    }


def test_sampled_mode_requires_sample_points():
    with pytest.raises(ValueError, match="sample_points"):
        PyAugmeconConfig(name="opts_test", mode="sampled")


def test_nadir_points_length_must_match_constrained_objectives():
    config = PyAugmeconConfig(name="opts_test", nadir_points=[1.0, 2.0])

    with pytest.raises(ValueError, match="nadir_points"):
        config.validate_against_model(num_objectives=2)


def test_sampled_scalar_points_normalized_to_list():
    config = PyAugmeconConfig(name="opts_test", mode="sampled", sample_points=10)

    config.validate_against_model(num_objectives=3)
    assert config.sample_points == [10]


def test_rejects_non_positive_objective_tolerance():
    with pytest.raises(ValueError, match="objective_tolerance"):
        PyAugmeconConfig(name="opts_test", objective_tolerance=0)


def test_rejects_invalid_work_distribution():
    with pytest.raises(Exception, match="work_distribution"):
        PyAugmeconConfig(name="opts_test", work_distribution="bad_value")


def test_rejects_invalid_flag_policy():
    with pytest.raises(Exception, match="flag_policy"):
        PyAugmeconConfig(name="opts_test", flag_policy="bad_value")


def test_sample_points_rejects_zero():
    with pytest.raises(ValueError, match="sample_points"):
        PyAugmeconConfig(name="opts_test", mode="sampled", sample_points=0)


def test_sample_points_rejects_negative_entries():
    with pytest.raises(ValueError, match="sample_points"):
        PyAugmeconConfig(name="opts_test", mode="sampled", sample_points=[5, -1])


def test_exact_mode_rejects_sample_points():
    with pytest.raises(ValueError, match="only valid in sampled mode"):
        PyAugmeconConfig(name="opts_test", mode="exact", sample_points=5)


def test_auto_work_distribution_depends_on_mode_and_workers():
    exact_multi = PyAugmeconConfig(name="test", mode="exact", workers=4)
    assert exact_multi.work_distribution == "outer_grid"

    exact_single = PyAugmeconConfig(name="test", mode="exact", workers=1)
    assert exact_single.work_distribution == "dynamic"

    sampled_multi = PyAugmeconConfig(
        name="test",
        mode="sampled",
        sample_points=10,
        workers=4,
    )
    assert sampled_multi.work_distribution == "dynamic"


def test_auto_flag_policy_depends_on_mode_and_workers():
    exact_multi = PyAugmeconConfig(name="test", mode="exact", workers=4)
    assert exact_multi.flag_policy == "shared"

    exact_single = PyAugmeconConfig(name="test", mode="exact", workers=1)
    assert exact_single.flag_policy == "local"

    sampled_multi = PyAugmeconConfig(
        name="test",
        mode="sampled",
        sample_points=10,
        workers=4,
    )
    assert sampled_multi.flag_policy == "local"
