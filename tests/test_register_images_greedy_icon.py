"""Unit tests for RegisterImagesGreedyICON.

The real-data behavioral-equivalence claim (RegisterImagesGreedyICON must
match the deleted RegisterTimeSeriesImages "Greedy_ICON" string path) is a
one-time manual check performed during implementation, not a permanent test
asset - see the implementation plan's Verification section. These tests
cover the class's wiring: it is a 2-stage RegisterImagesChain of the
expected backend types, accessible by name.
"""

from __future__ import annotations

import pytest

from monai_physio.register_images_chain import RegisterImagesChain
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_greedy_icon import RegisterImagesGreedyICON
from monai_physio.register_images_icon import RegisterImagesICON


def test_greedy_icon_is_a_two_stage_chain() -> None:
    """RegisterImagesGreedyICON wraps exactly [Greedy, ICON]."""
    registrar = RegisterImagesGreedyICON()
    assert isinstance(registrar, RegisterImagesChain)
    assert len(registrar.registrars) == 2
    assert isinstance(registrar.registrars[0], RegisterImagesGreedy)
    assert isinstance(registrar.registrars[1], RegisterImagesICON)


def test_greedy_icon_named_accessors() -> None:
    """.greedy/.icon return the same objects as positional registrars[0]/[1]."""
    registrar = RegisterImagesGreedyICON()
    assert registrar.greedy is registrar.registrars[0]
    assert registrar.icon is registrar.registrars[1]


def test_greedy_icon_stage_configuration_is_independent() -> None:
    """Configuring one stage must not affect the other."""
    registrar = RegisterImagesGreedyICON()
    registrar.greedy.set_number_of_iterations([30, 15, 7, 3])
    registrar.icon.set_number_of_iterations(20)

    assert registrar.greedy.number_of_iterations == [30, 15, 7, 3]
    assert registrar.icon.number_of_iterations == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
