import typing as tp

import numpy as np

from OgreInterface.data.universal_lennard_jones_parameters import (
    element_epsilons,
)
from OgreInterface.surface_matching.lj_surface_matcher.scatter_add import (
    scatter_add_bin,
)


class LJPotentialError(Exception):
    pass


class LJPotential:
    """
    Compute the lennard jones energy of a set of point charges inside a periodic box.
    Only works for periodic boundary conditions in all three spatial directions and orthorhombic boxes.
    Args:
        cutoff: cutoff radius
    """

    def __init__(
        self,
        cutoff: tp.Optional[float] = None,
    ):
        # Get the appropriate Coulomb constant
        self.cutoff = np.array(cutoff, dtype=np.float32)

    def forward(
        self,
        inputs: tp.Dict[str, np.ndarray],
    ) -> tp.Dict[str, np.ndarray]:
        idx_m = inputs["idx_m"]

        z = inputs["Z"]

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1
        r0s = inputs["r0s"]
        idx_m = inputs["idx_m"]
        # e_negs = inputs["e_negs"]

        idx_i_all = inputs["idx_i"]
        idx_j_all = inputs["idx_j"]

        R = inputs["R"]

        r_ij_all = R[idx_j_all] - R[idx_i_all] + inputs["offsets"]

        distances = np.sqrt(np.einsum("ij,ij->i", r_ij_all, r_ij_all))

        in_cutoff = distances <= self.cutoff
        idx_i = idx_i_all[in_cutoff]
        idx_j = idx_j_all[in_cutoff]
        epsilon_i = element_epsilons[z[idx_i]].astype(np.float32)
        epsilon_j = element_epsilons[z[idx_j]].astype(np.float32)

        epsilon_ij = np.sqrt(epsilon_i * epsilon_j).astype(np.float32)
        d_ij = distances[in_cutoff]

        r0_ij = r0s[idx_i] + r0s[idx_j]

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        y = self._lennard_jones(d_ij, epsilon_ij, r0_ij)

        y = scatter_add_bin(y, idx_i, dim_size=n_atoms)
        y = scatter_add_bin(y, idx_m, dim_size=n_molecules).reshape(-1)

        return y.astype(np.float32)

    def _lennard_jones(
        self,
        d_ij: np.ndarray,
        epsilon_ij: np.ndarray,
        r0_ij: np.ndarray,
    ) -> np.ndarray:
        energy = (
            4
            * epsilon_ij
            * ((0.25 * (r0_ij / d_ij) ** 12) - (0.5 * (r0_ij / d_ij) ** 6))
        )

        energy_cutoff = (
            4
            * epsilon_ij
            * (
                (0.25 * (r0_ij / self.cutoff) ** 12)
                - (0.5 * (r0_ij / self.cutoff) ** 6)
            )
        )

        return (energy - energy_cutoff).astype(np.float32)


if __name__ == "__main__":
    pass
