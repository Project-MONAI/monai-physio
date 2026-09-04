"""Composite registration: Greedy followed by ICON refinement.

This module provides the RegisterImagesGreedyICON class, a named
RegisterImagesChain of RegisterImagesGreedy followed by RegisterImagesICON.
"""

import logging

from .register_images_chain import RegisterImagesChain
from .register_images_greedy import RegisterImagesGreedy
from .register_images_icon import RegisterImagesICON


class RegisterImagesGreedyICON(RegisterImagesChain):
    """Greedy registration followed by ICON refinement, using Greedy's
    forward_transform to initialize ICON.

    Access the two stages by name via ``.greedy``/``.icon`` (e.g.
    ``RegisterImagesGreedyICON().greedy.set_number_of_iterations([30, 15, 7, 3])``)
    rather than positional ``registrars[0]``/``registrars[1]`` indexing.

    Example:
        >>> registrar = RegisterImagesGreedyICON()
        >>> registrar.greedy.set_number_of_iterations([30, 15, 7, 3])
        >>> registrar.icon.set_number_of_iterations(20)
        >>> registrar.set_fixed_image(fixed_image)
        >>> result = registrar.register(moving_image)
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the Greedy-then-ICON registration chain.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(
            [
                RegisterImagesGreedy(log_level=log_level),
                RegisterImagesICON(log_level=log_level),
            ],
            log_level=log_level,
        )

    @property
    def greedy(self) -> RegisterImagesGreedy:
        """The Greedy stage of this chain."""
        assert isinstance(self.registrars[0], RegisterImagesGreedy)
        return self.registrars[0]

    @property
    def icon(self) -> RegisterImagesICON:
        """The ICON stage of this chain."""
        assert isinstance(self.registrars[1], RegisterImagesICON)
        return self.registrars[1]
