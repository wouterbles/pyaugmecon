import pytest
from pyomo.environ import Var

from pyaugmecon import PyAugmecon
from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.solver.model import AUGMECON_RESERVED_NAMES
from tests.support.factories import make_config
from tests.support.models import two_objective_model


class _CaptureLogSink:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def log_sink():
    return _CaptureLogSink()


@pytest.mark.parametrize("reserved_name", sorted(AUGMECON_RESERVED_NAMES))
def test_reserved_component_name_conflict_raises(reserved_name: str):
    model = two_objective_model()
    setattr(model, reserved_name, Var())

    with pytest.raises(ValueError, match=reserved_name):
        PyAugmecon(model, make_config("component_conflict_test"))


def test_plain_dict_config_is_parsed_with_pydantic():
    py_augmecon = PyAugmecon(
        two_objective_model(),
        {
            "name": "dict_config_test",
            "progress_bar": False,
            "log_to_console": False,
        },
    )

    assert isinstance(py_augmecon.config, PyAugmeconConfig)
    assert py_augmecon.config.name == "dict_config_test"


def test_constructor_accepts_custom_log_sink(log_sink):
    PyAugmecon(
        two_objective_model(),
        {
            "name": "custom_log_sink_test",
            "progress_bar": False,
            "log_to_console": False,
        },
        log_sink=log_sink,
    )

    assert any("custom_log_sink_test" in message for message in log_sink.messages)


def test_solve_deactivates_all_objectives_automatically(mocker):
    model = two_objective_model()
    for objective_idx in model.obj_list:
        model.obj_list[objective_idx].activate()

    py_augmecon = PyAugmecon(model, make_config("objective_deactivation_test"))

    state = {}

    def stop_after_check():
        state["active"] = [
            py_augmecon.model.obj(i).active for i in py_augmecon.model.iter_obj
        ]
        raise RuntimeError("stop-test-after-objective-check")

    mocker.patch.object(py_augmecon.model, "min_to_max", side_effect=stop_after_check)

    with pytest.raises(RuntimeError, match="stop-test-after-objective-check"):
        py_augmecon.solve()

    assert state["active"] == [False, False]
