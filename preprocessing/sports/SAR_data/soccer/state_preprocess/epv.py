from pathlib import Path

import numpy as np

from constant import FIELD_LENGTH, FIELD_WIDTH


class EPV:
    def __init__(self, epv_grid_path: str | Path | None = None) -> None:
        if epv_grid_path is None:
            epv_grid_path = Path(__file__).parent / "EPV_grid.csv"
        self.epv_grid = np.loadtxt(epv_grid_path, delimiter=',')

    def calculate(self, x: float, y: float, attack_direction: int = 1) -> float:
        """
        Returns the EPV value at a given (x,y) location

        Parameters
        -----------
            position: Tuple containing the (x,y) pitch position
            attack_direction: Sets the attack direction (1: left->right, -1: right->left)

        Returrns
        -----------
            EPV value at input position

        """
        # make sure x is in the range [-FIELD_LENGTH/2, FIELD_LENGTH/2]
        # make sure y is in the range [-FIELD_WIDTH/2, FIELD_WIDTH/2]
        x = np.clip(x, -FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0)
        y = np.clip(y, -FIELD_WIDTH / 2.0, FIELD_WIDTH / 2.0)

        epv_grid = self.epv_grid if attack_direction == 1 else np.fliplr(self.epv_grid)
        ny, nx = epv_grid.shape
        dx = FIELD_LENGTH / float(nx)
        dy = FIELD_WIDTH / float(ny)
        ix = (x + FIELD_LENGTH / 2.0 - 0.0001) / dx
        iy = (y + FIELD_WIDTH / 2.0 - 0.0001) / dy
        return epv_grid[int(iy), int(ix)]
