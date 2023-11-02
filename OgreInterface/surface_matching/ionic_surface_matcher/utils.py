import typing as tp
import itertools
from os.path import join, dirname, split, abspath

from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from ase.data import chemical_symbols, covalent_radii
import numpy as np
import pandas as pd

from OgreInterface import data
from OgreInterface.surface_matching.ionic_surface_matcher.input_generator import (
    generate_input_dict,
    create_batch,
)

DATA_PATH = dirname(abspath(data.__file__))

IONIC_RADII_DF = pd.read_csv(join(DATA_PATH, "ionic_radii_data.csv"))


# def _add_r0s(struc):
#     r0s = []

#     for site in struc:
#         # atomic_number = site.specie.Z
#         atomic_number = site.properties["bulk_equivalent"]
#         if bool(site.properties["is_film"]):
#             r0s.append(r0_dict["film"][atomic_number])
#         else:
#             r0s.append(r0_dict["sub"][atomic_number])

#     struc.add_site_property("r0s", r0s)


# def _add_born_ns(struc):
#     ion_config_to_n_map = {
#         "1s1": 0.0,
#         "[He]": 5.0,
#         "[Ne]": 7.0,
#         "[Ar]": 9.0,
#         "[Kr]": 10.0,
#         "[Xe]": 12.0,
#     }
#     n_vals = {}

#     Zs = np.unique(struc.atomic_numbers)
#     for z in Zs:
#         element = Element(chemical_symbols[z])
#         ion_config = element.electronic_structure.split(".")[0]
#         n_val = ion_config_to_n_map[ion_config]
#         if _auto_determine_born_n:
#             n_vals[z] = n_val
#         else:
#             n_vals[z] = _born_n

#     ns = [n_vals[z] for z in struc.atomic_numbers]
#     struc.add_site_property("born_ns", ns)


def generate_base_inputs(
    structure: Structure,
    cutoff: float,
):
    inputs = generate_input_dict(
        structure=structure,
        cutoff=cutoff,
    )

    return inputs


def add_shifts_to_batch(
    batch_inputs: tp.Dict[str, np.ndarray],
    shifts: np.ndarray,
) -> None:
    if "is_film" in batch_inputs:
        n_atoms = batch_inputs["n_atoms"]
        all_shifts = np.repeat(
            shifts.astype(batch_inputs["R"].dtype), repeats=n_atoms, axis=0
        )
        all_shifts[~batch_inputs["is_film"]] *= 0.0
        batch_inputs["R"] += all_shifts
    else:
        raise "_add_shifts_to_batch should only be used on interfaces that have the is_film property"


def get_charges_from_structure(structure: Structure) -> tp.Dict[str, int]:
    """
    This function guesses the oxidation states from a given structure

    Args:
        structure: Input structure

    Returns:
        Dictionary of {symbol: charge}
    """
    oxi_guesses = structure.composition.oxi_state_guesses()

    if len(oxi_guesses) > 0:
        oxidation_states = oxi_guesses[0]
    else:
        unique_atomic_numbers = np.unique(structure.atomic_numbers)
        oxidation_states = {
            chemical_symbols[n]: 0 for n in unique_atomic_numbers
        }

    return oxidation_states


def get_equivalent_site_to_atomic_number_mapping(
    structure: Structure,
) -> tp.Dict[int, int]:
    """
    This function maps the equivalent sites to their atomic number

    Args:
        structure: Input structure (needs a bulk_equivalent site property)

    Returns:
        Dictionary of {bulk_equivalent: atomic_number}
    """
    # Add get bulk equivalent site property
    bulk_equiv = np.array(structure.site_properties["bulk_equivalent"])

    # Get the atomic numbers
    atomic_numbers = np.array(structure.atomic_numbers)

    # Create an dictionary mapping bulk equivalents to atomic numbers
    eq_to_Z = np.unique(np.c_[bulk_equiv, atomic_numbers], axis=0)
    eq_to_Z_dict = dict(zip(*eq_to_Z.T))

    return eq_to_Z_dict


def get_ionic_radii_from_structure(
    structure: Structure,
    charge_dict: tp.Dict[str, int],
    equiv_to_Z_dict: tp.Dict[int, int],
) -> tp.Dict[int, float]:
    """
    This function assigned atomic radii to each bulk equivalent site
    in a given structure

    Args:
        structure: Input structure (needs bulk_equivalent site property)
        charge_dict: Dictionary mapping {chemical_symbol: charge}
        equiv_to_Z_dict: Dictionary mapping {bulk_equivalent: atomic_number}

    Returns:
        Dictionary mapping {bulk_equivalent: ionic_radius}
    """
    # Create a copy of the structure to add oxidation states
    oxi_struc = structure.copy()

    # Add oxidation states using the charge_dict
    oxi_struc.add_oxidation_state_by_element(charge_dict)

    # Get unique bulk equivalent values
    unique_bulk_equiv = np.unique(oxi_struc.site_properties["bulk_equivalent"])

    # Get all combinations of bulk equivalent values
    # combos = itertools.combinations_with_replacement(unique_bulk_equiv, 2)

    # Create a dictionary to hold {(bulk_eq1, bulk_eq2): dist} values
    # neighbor_dict = {(eq1, eq2): None for (eq1, eq2) in combos}
    neighbor_dict = {}

    # Creaet an empty list to hold [[(bulk_eq1, bulk_eq2), dist]] values
    neighbor_list = []

    # Dictionary of ionic radii for each unique bulk equivalent site
    ionic_radii_dict = {eq: [] for eq in unique_bulk_equiv}

    # Dictionary of coordination numbers for each unique bulk equivalent site
    coordination_dict = {eq: [] for eq in unique_bulk_equiv}

    # Create a CrystallNN instance
    cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)

    # Loop through all sites in the structure to get the bonding environments
    for i, site in enumerate(oxi_struc.sites):
        # Get bulk equivalent of the center site
        site_equiv = site.properties["bulk_equivalent"]

        # Get nearest neighbor info dict
        info_dict = cnn.get_nn_info(oxi_struc, i)

        # The coordination of the site is the number of neighbors
        coordination_dict[site.properties["bulk_equivalent"]] = len(info_dict)

        # Loop through all the neighboring sites
        for neighbor in info_dict:
            # Get the bulk equivalent of the neighboring site
            neighbor_site_equiv = neighbor["site"].properties[
                "bulk_equivalent"
            ]

            # Get the bond vector in fractional coords
            frac_diff = site.frac_coords - neighbor["site"].frac_coords

            # Get the bond length
            bond_length = np.linalg.norm(
                oxi_struc.lattice.get_cartesian_coords(frac_diff)
            )

            # Get a tuple of the sorted bonding bulk equiv inds (0,1), (0,2)
            bonding_eqs = tuple(sorted([site_equiv, neighbor_site_equiv]))

            # Append the bonding equivs and bond length to the neighbor list
            neighbor_list.append([bonding_eqs, bond_length])

    # Sort the neighbor_list by bonding equiv tuples
    neighbor_list.sort(key=lambda x: x[0])

    # Group the neighbor_list by the bonding equiv tuples
    groups = itertools.groupby(neighbor_list, key=lambda x: x[0])

    # Loop through the groups for find the smallest bond length between
    # equivalent sites
    for group in groups:
        # Unpack the bond lengths from the group
        bond_lengths = list(zip(*group[1]))[1]

        # Add the minimum bond length to the neighbor_dict
        neighbor_dict[group[0]] = np.min(bond_lengths)

    # Loop through neighbor dict
    for (eq1, eq2), d in neighbor_dict.items():
        # Get the atomic number corresponding to eq1 and eq2
        z1 = equiv_to_Z_dict[eq1]
        z2 = equiv_to_Z_dict[eq2]

        # Get the chemical symbol from the atomic number
        s1 = chemical_symbols[z1]
        s2 = chemical_symbols[z2]

        # Get the charge from the charge_dict
        c1 = charge_dict[s1]
        c2 = charge_dict[s2]

        # Query the IONIC_RADII_DF to get potential radii based on
        # oxidation state and atomic number
        z1_df = IONIC_RADII_DF[
            (IONIC_RADII_DF["Atomic Number"] == z1)
            & (IONIC_RADII_DF["Oxidation State"] == c1)
        ]

        # If there is more than one option then get the radius that
        # best matches the coordination number of the given site
        if len(z1_df) > 0:
            # Get radius with closest coordination number
            z1_coords = z1_df["Coordination Number"].values
            z1_coord_diff = np.abs(z1_coords - coordination_dict[eq1])
            z1_coord_mask = z1_coord_diff == z1_coord_diff.min()
            z1_radii = z1_df[z1_coord_mask]

            if not pd.isna(z1_radii["Shannon"]).any():
                # If there is a shannon radius value use that
                d1 = z1_radii["Shannon"].values.mean() / 100
            else:
                # otherwise use the ML mean value
                d1 = z1_radii["ML Mean"].values.mean() / 100
        else:
            # If there are no entries use the covalent radius
            d1 = covalent_radii[z1]

        # Repeat the process for the other bonding species
        z2_df = IONIC_RADII_DF[
            (IONIC_RADII_DF["Atomic Number"] == z2)
            & (IONIC_RADII_DF["Oxidation State"] == c2)
        ]

        if len(z2_df) > 0:
            z2_coords = z2_df["Coordination Number"].values
            z2_coord_diff = np.abs(z2_coords - coordination_dict[eq2])
            z2_coord_mask = z2_coord_diff == z2_coord_diff.min()
            z2_radii = z2_df[z2_coord_mask]

            if not pd.isna(z2_radii["Shannon"]).any():
                d2 = z2_radii["Shannon"].values.mean() / 100
            else:
                d2 = z2_radii["ML Mean"].values.mean() / 100
        else:
            d2 = covalent_radii[z2]

        # Use the ionic radii of the two bonding species to partition the
        # bond length into ionic radii associated with each species
        radius_frac = d1 / (d1 + d2)

        # Extract the radius for species 1
        r0_1 = radius_frac * d

        # Extract the radius for species 2
        r0_2 = (1 - radius_frac) * d

        # Append the radius to the ionic_radii_dict[eq] list
        ionic_radii_dict[eq1].append(r0_1)
        ionic_radii_dict[eq2].append(r0_2)

    # Get the minimum radius from the list of ionic radii
    mean_radius_dict = {k: np.min(v) for k, v in ionic_radii_dict.items()}

    return mean_radius_dict


# def _get_r0s(sub, film, charge_dict):
#     sub_radii_dict, sub_eq_to_Z_dict = _get_neighborhood_info(
#         sub,
#         charge_dict["sub"],
#     )
#     film_radii_dict, film_eq_to_Z_dict = _get_neighborhood_info(
#         film,
#         charge_dict["film"],
#     )

#     r0_dict = {"film": film_radii_dict, "sub": sub_radii_dict}
#     eq_to_Z_dict = {"film": film_eq_to_Z_dict, "sub": sub_eq_to_Z_dict}

#     return r0_dict, eq_to_Z_dict
