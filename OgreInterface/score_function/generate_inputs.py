from ase.neighborlist import neighbor_list
from typing import Dict, List, Optional
from ase import Atoms
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import torch
import numpy as np
from OgreInterface.score_function.neighbors import TorchNeighborList
from OgreInterface.score_function.interface_neighbors import (
    TorchInterfaceNeighborList,
)
import time
from copy import deepcopy
from matscipy.neighbours import neighbour_list


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding
    Args:
        examples (list):
    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {"idx_i", "idx_j", "idx_i_triples"}
    # Atom triple indices must be treated separately
    idx_triple_keys = {"idx_j_triples", "idx_k_triples"}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch["n_atoms"], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch["n_atoms"], dim=0
    )
    coll_batch["idx_m"] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d["idx_j"].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def create_batch(
    inputs: Dict,
    batch_size: int,
):
    batch_inputs = {}
    n_atoms = int(inputs["n_atoms"][0])
    offsets = torch.arange(0, batch_size, 1)
    idx_m = offsets.repeat_interleave(n_atoms)
    batch_inputs["idx_m"] = idx_m

    for k, v in inputs.items():
        repeat_val = [1] * len(v.shape)
        repeat_val[0] = batch_size
        repeat_val = tuple(repeat_val)
        if "idx" in k:
            idx_len = len(v)
            idx_offsets = n_atoms * offsets
            batch_offsets = idx_offsets.repeat_interleave(idx_len)
            batch_idx = v.repeat(batch_size)
            batch_idx += batch_offsets
            batch_inputs[k] = batch_idx.to(dtype=v.dtype)
        else:
            batch_val = v.repeat(repeat_val)
            batch_inputs[k] = batch_val.to(dtype=v.dtype)

    for k, v in batch_inputs.items():
        if "float" in str(v.dtype):
            batch_inputs[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            batch_inputs[k] = v.to(dtype=torch.long)

        # print(k, batch_inputs[k].dtype)

    return batch_inputs


def generate_input_dict(
    structure: Structure,
    cutoff: float,
    interface: bool = False,
) -> Dict:

    if interface:
        tn = TorchInterfaceNeighborList(cutoff=cutoff)
    else:
        tn = TorchNeighborList(cutoff=cutoff)

    site_props = structure.site_properties

    is_film = torch.tensor(site_props["is_film"], dtype=torch.long)
    R = torch.from_numpy(structure.cart_coords)
    cell = torch.from_numpy(deepcopy(structure.lattice.matrix))

    e_negs = torch.Tensor([s.specie.X for s in structure])

    if interface:
        pbc = torch.Tensor([True, True, False]).to(dtype=torch.bool)
    else:
        pbc = torch.Tensor([True, True, True]).to(dtype=torch.bool)

    input_dict = {
        "n_atoms": torch.tensor([len(structure)]),
        "Z": torch.tensor(structure.atomic_numbers, dtype=torch.long),
        "R": R,
        "cell": cell,
        "pbc": pbc,
        "is_film": is_film,
        "e_negs": e_negs,
    }

    if "charges" in site_props:
        charges = torch.tensor(site_props["charges"])
        input_dict["partial_charges"] = charges

    if "born_ns" in site_props:
        ns = torch.tensor(site_props["born_ns"])
        input_dict["born_ns"] = ns

    tn.forward(inputs=input_dict)
    input_dict["cell"] = input_dict["cell"].view(-1, 3, 3)
    input_dict["pbc"] = input_dict["pbc"].view(-1, 3)

    for k, v in input_dict.items():
        if "float" in str(v.dtype):
            input_dict[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            input_dict[k] = v.to(dtype=torch.long)

    return input_dict


def generate_input_dict_matscipy(
    structure: Structure,
    cutoff: float,
    interface: bool = False,
) -> Dict:
    site_props = structure.site_properties

    is_film = torch.tensor(site_props["is_film"], dtype=torch.long)
    R = torch.from_numpy(structure.cart_coords)
    cell = torch.from_numpy(deepcopy(structure.lattice.matrix))

    e_negs = torch.Tensor([s.specie.X for s in structure])

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if interface:
        pbc = torch.Tensor([True, True, False]).to(dtype=torch.bool)
        atoms.set_pbc([True, True, False])
    else:
        pbc = torch.Tensor([True, True, True]).to(dtype=torch.bool)

    idx_i, idx_j, frac_offsets = neighbour_list(
        "ijS",
        atoms=atoms,
        cutoff=cutoff,
    )
    offsets = frac_offsets.dot(atoms.cell)

    input_dict = {
        "n_atoms": torch.tensor([len(structure)]),
        "Z": torch.tensor(structure.atomic_numbers, dtype=torch.long),
        "R": R,
        "cell": cell.view(-1, 3, 3),
        "pbc": pbc.view(-1, 3),
        "is_film": is_film,
        "e_negs": e_negs,
        "idx_i": torch.from_numpy(idx_i),
        "idx_j": torch.from_numpy(idx_j),
        "offsets": torch.from_numpy(offsets),
    }

    if "charges" in site_props:
        charges = torch.tensor(site_props["charges"])
        input_dict["partial_charges"] = charges

    if "born_ns" in site_props:
        ns = torch.tensor(site_props["born_ns"])
        input_dict["born_ns"] = ns

    for k, v in input_dict.items():
        if "float" in str(v.dtype):
            input_dict[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            input_dict[k] = v.to(dtype=torch.long)

    return input_dict


def generate_dict_torch_bu(
    atoms: Atoms,
    shifts: np.ndarray,
    cutoff: float,
    interface: bool = False,
    ns_dict: Optional[Dict[str, float]] = None,
    charge_dict: Optional[Dict[str, float]] = None,
    z_shift: float = 15.0,
    z_periodic: bool = False,
) -> Dict:

    if interface:
        tn = TorchInterfaceNeighborList(cutoff=cutoff)
    else:
        tn = TorchNeighborList(cutoff=cutoff)

    neighbor_inputs = {}
    inputs_batch = []

    for at_idx, shift in enumerate(shifts):
        is_film = torch.from_numpy(
            atoms.get_array("is_film", copy=True).astype(int)
        )
        R = torch.from_numpy(atoms.get_positions())
        z_positions = np.copy(atoms.get_positions())
        z_positions[atoms.get_array("is_film"), -1] += z_shift

        R_z = torch.from_numpy(z_positions)
        cell = torch.from_numpy(atoms.get_cell().array)
        recip_cell = torch.from_numpy(atoms.get_reciprocal_cell().array)

        e_negs = torch.Tensor(
            [Element(s).X for s in atoms.get_chemical_symbols()]
        )

        if interface:
            pbc = torch.Tensor([True, True, False]).to(dtype=torch.bool)
        else:
            pbc = torch.Tensor([True, True, True]).to(dtype=torch.bool)

        input_dict = {
            "n_atoms": torch.tensor([atoms.get_global_number_of_atoms()]),
            "Z": torch.from_numpy(atoms.get_atomic_numbers()),
            "R": R,
            "R_z": R_z,
            "cell": cell,
            "recip_cell": recip_cell.view(-1, 3, 3),
            "pbc": pbc,
            "is_film": is_film,
            "e_negs": e_negs,
            "shift": torch.from_numpy(shift).view(-1, 3),
        }

        if charge_dict is not None:
            charges = torch.Tensor(
                [charge_dict[s] for s in atoms.get_chemical_symbols()]
            )
            # ns = torch.Tensor(
            #     [ns_dict[s] for s in atoms.get_chemical_symbols()]
            # )
            input_dict["partial_charges"] = charges
            # input_dict["ns"] = ns

        if at_idx == 0:
            s = time.time()
            tn.forward(inputs=input_dict)
            neighbor_inputs["idx_i"] = input_dict["idx_i"]
            neighbor_inputs["idx_j"] = input_dict["idx_j"]
            neighbor_inputs["offsets"] = input_dict["offsets"]
            # print("Neighbor Time: ", time.time() - s)
        else:
            input_dict.update(neighbor_inputs)

        input_dict["cell"] = input_dict["cell"].view(-1, 3, 3)
        input_dict["pbc"] = input_dict["pbc"].view(-1, 3)

        inputs_batch.append(input_dict)

    s = time.time()
    inputs = _atoms_collate_fn(inputs_batch)
    # print("Collate:", time.time() - s)

    for k, v in inputs.items():
        if "float" in str(v.dtype):
            inputs[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            inputs[k] = v.to(dtype=torch.long)

    return inputs


def generate_dict_torch_old(
    atoms: List[Atoms],
    cutoff: float,
    interface: bool = False,
    ns_dict: Optional[Dict[str, float]] = None,
    charge_dict: Optional[Dict[str, float]] = None,
    z_shift: float = 15.0,
    z_periodic: bool = False,
) -> Dict:

    if interface:
        tn = TorchInterfaceNeighborList(cutoff=cutoff)
    else:
        tn = TorchNeighborList(cutoff=cutoff)

    neighbor_inputs = {}
    inputs_batch = []

    for at_idx, atom in enumerate(atoms):

        is_film = torch.from_numpy(
            atom.get_array("is_film", copy=True).astype(int)
        )
        R = torch.from_numpy(atom.get_positions())
        z_positions = np.copy(atom.get_positions())
        z_positions[atom.get_array("is_film"), -1] += z_shift

        R_z = torch.from_numpy(z_positions)
        cell = torch.from_numpy(atom.get_cell().array)

        e_negs = torch.Tensor(
            [Element(s).X for s in atom.get_chemical_symbols()]
        )

        if z_periodic:
            pbc = torch.Tensor([True, True, True]).to(dtype=torch.bool)
        else:
            pbc = torch.Tensor([True, True, False]).to(dtype=torch.bool)

        input_dict = {
            "n_atoms": torch.tensor([atom.get_global_number_of_atoms()]),
            "Z": torch.from_numpy(atom.get_atomic_numbers()),
            "R": R,
            "R_z": R_z,
            "cell": cell,
            "pbc": pbc,
            "is_film": is_film,
            "e_negs": e_negs,
        }

        if charge_dict is not None:
            charges = torch.Tensor(
                [charge_dict[s] for s in atom.get_chemical_symbols()]
            )
            ns = torch.Tensor(
                [ns_dict[s] for s in atom.get_chemical_symbols()]
            )
            input_dict["partial_charges"] = charges
            input_dict["ns"] = ns

        if at_idx == 0:
            tn.forward(inputs=input_dict)
            neighbor_inputs["idx_i"] = input_dict["idx_i"]
            neighbor_inputs["idx_j"] = input_dict["idx_j"]
            neighbor_inputs["offsets"] = input_dict["offsets"]
        else:
            input_dict.update(neighbor_inputs)

        input_dict["cell"] = input_dict["cell"].view(-1, 3, 3)
        input_dict["pbc"] = input_dict["pbc"].view(-1, 3)

        inputs_batch.append(input_dict)

    inputs = _atoms_collate_fn(inputs_batch)

    for k, v in inputs.items():
        if "float" in str(v.dtype):
            inputs[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            inputs[k] = v.to(dtype=torch.long)

    return inputs


if __name__ == "__main__":
    from ase.build import bulk

    InAs = bulk("InAs", crystalstructure="zincblende", a=5.6)
    charge_dict = {"In": 0.0, "As": 0.0}
    inputs = generate_dict_torch([InAs], cutoff=10.0, charge_dict=charge_dict)
    print(inputs["n_atoms"])
