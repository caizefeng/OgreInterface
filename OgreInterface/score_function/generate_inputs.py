from typing import Dict, List, Optional
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from copy import deepcopy
from matscipy.neighbours import neighbour_list


def create_batch(
    inputs: Dict[str, np.ndarray],
    batch_size: int,
):
    batch_inputs = {}
    n_atoms = int(inputs["n_atoms"][0])
    offsets = np.arange(batch_size)
    idx_m = np.repeat(offsets, n_atoms)
    batch_inputs["idx_m"] = idx_m

    for k, v in inputs.items():
        if "float" in str(v.dtype):
            new_dtype = np.float32
        else:
            new_dtype = v.dtype

        repeat_val = [1] * len(v.shape)
        repeat_val[0] = batch_size
        repeat_val = tuple(repeat_val)
        if "idx" in k:
            idx_len = len(v)
            idx_offsets = n_atoms * offsets
            batch_offsets = np.repeat(idx_offsets, idx_len)
            batch_idx = np.tile(v, repeat_val)
            batch_idx += batch_offsets
            batch_inputs[k] = batch_idx.astype(new_dtype)
        else:
            batch_val = np.tile(v, repeat_val)
            batch_inputs[k] = batch_val.astype(new_dtype)

    return batch_inputs


# def generate_input_dict(
#     structure: Structure,
#     cutoff: float,
#     interface: bool = False,
# ) -> Dict:

#     if interface:
#         tn = TorchInterfaceNeighborList(cutoff=cutoff)
#     else:
#         tn = TorchNeighborList(cutoff=cutoff)

#     site_props = structure.site_properties

#     is_film = torch.tensor(site_props["is_film"], dtype=torch.long)
#     R = torch.from_numpy(structure.cart_coords)
#     cell = torch.from_numpy(deepcopy(structure.lattice.matrix))

#     e_negs = torch.Tensor([s.specie.X for s in structure])

#     if interface:
#         pbc = torch.Tensor([True, True, False]).to(dtype=torch.bool)
#     else:
#         pbc = torch.Tensor([True, True, True]).to(dtype=torch.bool)

#     input_dict = {
#         "n_atoms": torch.tensor([len(structure)]),
#         "Z": torch.tensor(structure.atomic_numbers, dtype=torch.long),
#         "R": R,
#         "cell": cell,
#         "pbc": pbc,
#         "is_film": is_film,
#         "e_negs": e_negs,
#     }

#     if "charge" in site_props:
#         charges = torch.tensor(site_props["charge"])
#         input_dict["partial_charges"] = charges

#     if "born_ns" in site_props:
#         ns = torch.tensor(site_props["born_ns"])
#         input_dict["born_ns"] = ns

#     tn.forward(inputs=input_dict)
#     input_dict["cell"] = input_dict["cell"].view(-1, 3, 3)
#     input_dict["pbc"] = input_dict["pbc"].view(-1, 3)

#     for k, v in input_dict.items():
#         if "float" in str(v.dtype):
#             input_dict[k] = v.to(dtype=torch.float32)
#         if "idx" in k:
#             input_dict[k] = v.to(dtype=torch.long)

#     return input_dict


def generate_input_dict(
    structure: Structure,
    cutoff: float,
    interface: bool = False,
) -> Dict:
    site_props = structure.site_properties

    is_film = np.array(site_props["is_film"]).astype(bool)
    R = structure.cart_coords
    cell = deepcopy(structure.lattice.matrix)

    e_negs = np.array([s.specie.X for s in structure])

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if interface:
        pbc = np.array([True, True, False])
        atoms.set_pbc([True, True, False])
    else:
        pbc = np.array([True, True, True])

    idx_i, idx_j, frac_offsets = neighbour_list(
        "ijS",
        atoms=atoms,
        cutoff=cutoff,
    )
    offsets = frac_offsets.dot(atoms.cell)

    input_dict = {
        "n_atoms": np.array([len(structure)]),
        "Z": np.array(structure.atomic_numbers).astype(int),
        "R": R,
        "cell": cell.reshape(-1, 3, 3),
        "pbc": pbc.reshape(-1, 3),
        "is_film": is_film,
        "e_negs": e_negs,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "offsets": offsets,
    }

    if "charge" in site_props:
        charges = np.array(site_props["charge"])
        input_dict["partial_charges"] = charges

    if "born_ns" in site_props:
        ns = np.array(site_props["born_ns"])
        input_dict["born_ns"] = ns

    if "r0s" in site_props:
        r0s = np.array(site_props["r0s"])
        input_dict["r0s"] = r0s

    return input_dict


if __name__ == "__main__":
    from ase.build import bulk

    InAs = bulk("InAs", crystalstructure="zincblende", a=5.6)
    charge_dict = {"In": 0.0, "As": 0.0}
