from typing import Dict, Optional, Tuple
import time
import numpy as np
from scipy.special import erfc
from OgreInterface.score_function.scatter import scatter_add, scatter_add_bin


class IonicPotentialError(Exception):
    pass


class IonicShiftedForcePotential:
    """
    Compute the Coulomb energy of a set of point charges inside a periodic box.
    Only works for periodic boundary conditions in all three spatial directions and orthorhombic boxes.
    Args:
        alpha (float): Ewald alpha.
        k_max (int): Number of lattice vectors.
        charges_key (str): Key of partial charges in the input batch.
    """

    def __init__(
        self,
        cutoff: Optional[float] = None,
    ):
        # Get the appropriate Coulomb constant
        self.ke = np.array(14.3996, dtype=np.float32)
        self.cutoff = np.array(cutoff, dtype=np.float32)

    def forward(
        self,
        inputs: Dict[str, np.ndarray],
        # shift: np.ndarray,
        constant_coulomb_contribution: Optional[np.ndarray] = None,
        constant_born_contribution: Optional[np.ndarray] = None,
        # r0_array: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        q = inputs["partial_charges"]
        idx_m = inputs["idx_m"]

        n_atoms = q.shape[0]
        n_molecules = int(idx_m[-1]) + 1
        z = inputs["Z"]
        ns = inputs["born_ns"]
        r0s = inputs["r0s"]
        idx_m = inputs["idx_m"]
        e_negs = inputs["e_negs"]

        idx_i_all = inputs["idx_i"]
        idx_j_all = inputs["idx_j"]

        R = inputs["R"]

        r_ij_all = R[idx_j_all] - R[idx_i_all] + inputs["offsets"]

        distances = np.sqrt(np.einsum("ij,ij->i", r_ij_all, r_ij_all))

        in_cutoff = distances <= self.cutoff
        idx_i = idx_i_all[in_cutoff]
        idx_j = idx_j_all[in_cutoff]
        d_ij = distances[in_cutoff]

        r0_ij = r0s[idx_i] + r0s[idx_j]
        n_ij = (ns[idx_i] + ns[idx_j]) / 2
        q_ij = (q[idx_i] * q[idx_j]).astype(np.float32)
        e_diff_ij = 0.5 + (np.abs(e_negs[idx_i] - e_negs[idx_j]) / (2 * 3.19))
        zero_charge_mask = q_ij == 0
        q_ij[zero_charge_mask] -= e_diff_ij[zero_charge_mask]

        B_ij = -self._calc_B(r0_ij=r0_ij, n_ij=n_ij, q_ij=q_ij)

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        y_dsf, y_dsf_self = self._damped_shifted_force(d_ij, q_ij, q)

        y_dsf = scatter_add_bin(y_dsf, idx_i, dim_size=n_atoms)
        y_dsf = scatter_add_bin(y_dsf, idx_m, dim_size=n_molecules)

        if constant_coulomb_contribution is not None:
            y_dsf += np.tile(constant_coulomb_contribution, n_molecules)

        y_dsf_self = scatter_add_bin(y_dsf_self, idx_m, dim_size=n_molecules)
        y_coulomb = 0.5 * self.ke * (y_dsf - y_dsf_self).reshape(-1)

        y_born = self._born(d_ij, n_ij, B_ij)

        y_born = scatter_add_bin(y_born, idx_i, dim_size=n_atoms)
        y_born = scatter_add_bin(y_born, idx_m, dim_size=n_molecules)

        if constant_born_contribution is not None:
            y_born += np.tile(constant_born_contribution, n_molecules) / (
                0.5 * self.ke
            )

        y_born = 0.5 * self.ke * y_born.reshape(-1)

        y_energy = y_coulomb + y_born

        return (
            y_energy.astype(np.float32),
            y_coulomb.astype(np.float32),
            y_born.astype(np.float32),
            y_dsf,
            None,
        )

    def _calc_B(self, r0_ij, n_ij, q_ij):
        alpha = np.array(0.2, dtype=np.float32)
        pi = np.array(np.pi, dtype=np.float32)
        pre_factor = ((r0_ij ** (n_ij + 1)) * np.abs(q_ij)) / n_ij
        term1 = erfc(alpha * r0_ij) / (r0_ij**2).astype(np.float32)
        term2 = (2 * alpha / np.sqrt(pi)) * (
            np.exp(-(alpha**2) * (r0_ij**2)) / r0_ij
        )
        term3 = erfc(alpha * self.cutoff).astype(np.float32) / self.cutoff**2
        term4 = (
            (2 * alpha / np.sqrt(pi))
            * (np.exp(-(alpha**2) * (self.cutoff**2)) / self.cutoff)
        ).astype(np.float32)

        B_ij = pre_factor * (-term1 - term2 + term3 + term4)

        return B_ij

    def _born(self, d_ij: np.ndarray, n_ij: np.ndarray, B_ij: np.ndarray):
        return B_ij * ((1 / (d_ij**n_ij)) - (1 / (self.cutoff**n_ij)))

    def _damped_shifted_force(
        self, d_ij: np.ndarray, q_ij: np.ndarray, q: np.ndarray
    ):
        alpha = 0.2

        self_energy = (
            (erfc(alpha * self.cutoff) / self.cutoff)
            + (alpha / np.sqrt(np.pi))
        ) * (q**2)

        energies = q_ij * (
            (erfc(alpha * d_ij) / d_ij)
            - (erfc(alpha * self.cutoff) / self.cutoff)
            + (
                (
                    (erfc(alpha * self.cutoff) / self.cutoff**2)
                    + (
                        (2 * alpha / np.sqrt(np.pi))
                        * (
                            np.exp(-(alpha**2) * (self.cutoff**2))
                            / self.cutoff
                        )
                    )
                )
                * (d_ij - self.cutoff)
            )
        )

        return energies.astype(np.float32), self_energy.astype(np.float32)


if __name__ == "__main__":
    pass
