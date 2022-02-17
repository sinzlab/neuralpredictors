from typing import Callable, Protocol

import pytest

from neuralpredictors.layers.cores.conv2d import Stacked2dCore


class CreateCore(Protocol):
    def __call__(self, *, final_nonlinearity: bool, linear: bool) -> Stacked2dCore:
        ...


@pytest.fixture
def create_core() -> CreateCore:
    def _create_core(*, final_nonlinearity: bool, linear: bool) -> Stacked2dCore:
        return Stacked2dCore(  # type: ignore[no-untyped-call]
            input_channels=10,
            hidden_channels=10,
            input_kern=5,
            hidden_kern=5,
            layers=3,
            final_nonlinearity=final_nonlinearity,
            linear=linear,
        )

    return _create_core


def test_final_nonlinearity_is_true(create_core: CreateCore) -> None:
    core = create_core(final_nonlinearity=True, linear=False)
    assert hasattr(core.features[-1], "nonlin")


def test_final_nonlinearity_is_false(create_core: CreateCore) -> None:
    core = create_core(final_nonlinearity=False, linear=False)
    assert not hasattr(core.features[-1], "nonlin")


def test_intermdiate_layers_if_final_nonlinearity_is_false(create_core: CreateCore) -> None:
    core = create_core(final_nonlinearity=False, linear=False)
    assert hasattr(core.features[-2], "nonlin")


def test_intermediate_layer_if_linear_is_true(create_core: CreateCore) -> None:
    core = create_core(final_nonlinearity=False, linear=True)
    assert not hasattr(core.features[-2], "nonlin")
