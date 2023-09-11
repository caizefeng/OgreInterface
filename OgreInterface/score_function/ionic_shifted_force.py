from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
import time


class IonicPotentialError(Exception):
    pass


class IonicShiftedForcePotential(nn.Module):
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
        super(IonicShiftedForcePotential, self).__init__()

        # Get the appropriate Coulomb constant
        ke = 14.3996
        self.register_buffer("ke", torch.Tensor([ke]))

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        shift: torch.Tensor,
        r0_array: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # print("Non-zeros =", (r0_dict != 0.0).sum())
        # r0_array = torch.from_numpy(r0_array)

        q = inputs["partial_charges"].squeeze(-1)
        idx_m = inputs["idx_m"]

        n_atoms = q.shape[0]
        n_molecules = int(idx_m[-1]) + 1
        z = inputs["Z"]
        ns = inputs["born_ns"]
        idx_m = inputs["idx_m"]

        idx_i_all = inputs["idx_i"]
        idx_j_all = inputs["idx_j"]
        is_film = inputs["is_film"]
        is_film_bool = torch.clone(is_film).to(dtype=bool)
        is_sub_bool = torch.logical_not(is_film_bool)

        R = inputs["R"]
        # R.requires_grad_()

        shift.requires_grad_()
        shifts = torch.repeat_interleave(
            shift, repeats=inputs["n_atoms"], dim=0
        )
        shifts[is_sub_bool] *= 0.0
        # shifts.requires_grad_()

        # s = time.time()
        R_shift = R + shifts
        r_ij_all = R_shift[idx_j_all] - R_shift[idx_i_all] + inputs["offsets"]
        # print("Beginning:", round(time.time() - s, 4))

        # TODO make this fast

        distances = torch.norm(r_ij_all, dim=1)
        in_cutoff = torch.nonzero(distances < self.cutoff, as_tuple=False)
        pair_index = in_cutoff.squeeze()
        idx_i = idx_i_all[pair_index]
        idx_j = idx_j_all[pair_index]
        offsets = inputs["offsets"][pair_index]

        is_film_i = is_film[idx_i]
        is_film_j = is_film[idx_j]

        r_ij = R_shift[idx_j] - R_shift[idx_i] + offsets
        # print("Total Number of rs", len(r_ij))
        # print(
        #     "Number of Constant rs",
        #     (r_ij[is_film_i][:, -1] > 0.0).sum()
        #     + (r_ij[~is_film_i][:, -1] < 0.0).sum(),
        # )

        # print("is_film_i =", is_film_i.dtype)
        # print("is_film_j =", is_film_j.dtype)
        # print("is_film_i + is_film_j =", (is_film_i + is_film_j).dtype)

        # s = time.time()
        r0_ij = r0_array[is_film_i + is_film_j, z[idx_i], z[idx_j]]
        n_ij = (ns[idx_i] + ns[idx_j]) / 2
        # ns_dict[z[idx_i], z[idx_j]]

        # print("R_ij all:", time.time() - s)

        # r0_ij = torch.from_numpy(r0_ij).to(dtype=torch.float32)
        # n_ij = torch.from_numpy(n_ij).to(dtype=torch.float32)

        d_ij = torch.norm(r_ij, dim=1)
        q_ij = q[idx_i] * q[idx_j]

        # B_ij = (torch.abs(q_ij) * (r0_ij ** (n_ij - 1.0))) / n_ij
        B_ij = -self._calc_B(r0_ij=r0_ij, n_ij=n_ij, q_ij=q_ij)

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        y_dsf, y_dsf_self = self._damped_shifted_force(d_ij, q_ij, q)
        y_dsf = scatter_add(y_dsf, idx_i, dim_size=n_atoms)
        y_dsf = scatter_add(y_dsf, idx_m, dim_size=n_molecules)
        y_dsf_self = scatter_add(y_dsf_self, idx_m, dim_size=n_molecules)
        y_coulomb = 0.5 * self.ke * torch.squeeze(y_dsf - y_dsf_self, -1)
        # print("Coulomb:", time.time() - s)

        # s = time.time()
        y_born = self._born(d_ij, n_ij, B_ij)
        y_born = scatter_add(y_born, idx_i, dim_size=n_atoms)
        y_born = scatter_add(y_born, idx_m, dim_size=n_molecules)
        y_born = 0.5 * self.ke * torch.squeeze(y_born, -1)
        # print("Born:", time.time() - s)

        y_energy = y_coulomb + y_born

        # s = time.time()
        go = [torch.ones_like(y_energy)]
        grads = grad([y_energy], [shift], grad_outputs=go, create_graph=False)
        shift_gradient = grads[0]
        # dEdR[is_sub_bool] *= 0.0

        # go = [torch.ones_like(y_energy)]
        # grads = grad(
        #     [y_energy], [r0_dict], grad_outputs=go, create_graph=False
        # )
        # dEdR0 = grads[0]

        # dEdR0.detach_()

        # dEdR0 = dEdR0.numpy()
        # print(dEdR0.shape)
        # print(np.round(dEdR0[dEdR0 != 0.0], 4).tolist())

        # film_force = scatter_add(dEdR, idx_m, dim_size=n_molecules)
        # film_force_norm = torch.squeeze(torch.norm(film_force, dim=1) ** 2)

        # f_go = [torch.ones_like(film_force_norm)]
        # film_norm_grad = grad([film_force_norm], [shift], grad_outputs=f_go)[0]
        # film_norm_grad = torch.squeeze(film_norm_grad)

        R_shift.detach_()
        shifts.detach_()
        shift.detach_()

        return (
            y_energy.detach().numpy(),
            y_coulomb.detach().numpy(),
            y_born.detach().numpy(),
            # y_energy.numpy(),
            # y_coulomb.numpy(),
            # y_born.numpy(),
            shift_gradient,
            None,
        )

    def _calc_B(self, r0_ij, n_ij, q_ij):
        alpha = 0.2
        pi = torch.tensor(np.pi)
        pre_factor = ((r0_ij ** (n_ij + 1)) * torch.abs(q_ij)) / n_ij
        term1 = torch.erfc(alpha * r0_ij) / (r0_ij**2)
        term2 = (2 * alpha / torch.sqrt(pi)) * (
            torch.exp(-(alpha**2) * (r0_ij**2)) / r0_ij
        )
        term3 = torch.erfc(alpha * self.cutoff) / self.cutoff**2
        term4 = (2 * alpha / torch.sqrt(pi)) * (
            torch.exp(-(alpha**2) * (self.cutoff**2)) / self.cutoff
        )

        B_ij = pre_factor * (-term1 - term2 + term3 + term4)

        return B_ij

    def _born(
        self, d_ij: torch.Tensor, n_ij: torch.Tensor, B_ij: torch.Tensor
    ):
        return B_ij * ((1 / (d_ij**n_ij)) - (1 / (self.cutoff**n_ij)))

    def _damped_shifted_force(
        self, d_ij: torch.Tensor, q_ij: torch.Tensor, q: torch.Tensor
    ):
        alpha = 0.2

        self_energy = (
            (torch.erfc(alpha * self.cutoff) / self.cutoff)
            + (alpha / torch.sqrt(torch.tensor(np.pi)))
        ) * (q**2)

        energies = q_ij * (
            (torch.erfc(alpha * d_ij) / d_ij)
            - (torch.erfc(alpha * self.cutoff) / self.cutoff)
            + (
                (
                    (torch.erfc(alpha * self.cutoff) / self.cutoff**2)
                    + (
                        (2 * alpha / torch.sqrt(torch.tensor(np.pi)))
                        * (
                            torch.exp(-(alpha**2) * (self.cutoff**2))
                            / self.cutoff
                        )
                    )
                )
                * (d_ij - self.cutoff)
            )
        )

        return energies, self_energy

    def _wolf_old(
        self, d_ij: torch.Tensor, q_ij: torch.Tensor, q: torch.Tensor
    ):
        alpha = 0.2
        energies = q_ij * (
            (torch.erfc(alpha * d_ij) / d_ij)
            - (torch.erfc(alpha * self.cutoff) / self.cutoff)
        )

        self_energy = (
            (torch.erfc(alpha * self.cutoff) / self.cutoff)
            + (alpha / torch.sqrt(torch.tensor(np.pi)))
        ) * (q**2)

        return energies, self_energy


if __name__ == "__main__":
    pass
