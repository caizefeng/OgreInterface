from __future__ import annotations
import numpy as np
import copy
from functools import reduce
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import DummySpecies
import itertools
import functools
import math
import collections
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
import networkx as nx
import spglib


def _get_colored_molecules(struc, output):
    colored_struc = struc.copy()
    for site in colored_struc:
        if "dummy_species" in site.properties:
            ds = site.properties["dummy_species"]
        else:
            ds = site.species.Z

        site.species = DummySpecies(symbol=f"Q{chr(ds - 22 + ord('a'))}")

    colored_struc.sort()
    Poscar(colored_struc).write_file(output)


def apply_strain_matrix(
    structure: Structure, strain_matrix: np.ndarray
) -> Structure:
    """
    This function applies a strain matrix to a structure to match it to another lattice
    i.e. straining a film to match with the substrate material. The strain_matrix can be calculated
    by the following equation:
        strain_matrix = np.linalg.inv(old_lattice_matrix) @ new_lattice_matrix
    """
    new_matrix = structure.lattice.matrix @ strain_matrix

    strained_structure = Structure(
        lattice=Lattice(new_matrix),
        species=structure.species,
        coords=structure.frac_coords,
        to_unit_cell=True,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )

    return strained_structure


def spglib_standardize(
    structure: Structure,
    to_primitive: bool = False,
    no_idealize: bool = False,
) -> Structure:
    """
    This function standardized a given structure using the spglib library

    Args:
        structure: Input pymatgen Structure
        to_primitive: Determines if the structure should be converted to it's primitive unit cell
        no_idealize: Determines if the lattice vectors should be idealized
            (i.e. rotate a cubic structure so \\vec{a} points along the [1, 0, 0] cartesian direction)

    Returns:
        The standardized structure in the form of a pymatgen Structure object
    """
    init_lattice = structure.lattice.matrix
    init_positions = structure.frac_coords
    init_numbers = np.array(structure.atomic_numbers)
    init_cell = (init_lattice, init_positions, init_numbers)

    (
        standardized_lattice,
        standardized_positions,
        standardized_numbers,
    ) = spglib.standardize_cell(
        init_cell,
        to_primitive=to_primitive,
        no_idealize=no_idealize,
    )

    standardized_structure = Structure(
        lattice=Lattice(standardized_lattice),
        species=standardized_numbers,
        coords=standardized_positions,
        to_unit_cell=True,
        coords_are_cartesian=False,
    )

    return standardized_structure


def apply_op_to_mols(struc, op):
    for site in struc:
        mol = site.properties["molecules"]
        op_mol = mol.copy()
        op_mol.translate_sites(range(len(mol)), site.coords)
        op_mol.apply_operation(op)
        centered_mol = op_mol.get_centered_molecule()
        site.properties["molecules"] = centered_mol


def replace_molecules_with_atoms(s: Structure) -> Structure:
    # Create a structure graph so we can extract the molecules
    struc_graph = StructureGraph.with_local_env_strategy(s, JmolNN())

    # Find the center of masses of all the molecules in the unit cell
    # We can do this similar to how the get_subgraphs_as_molecules()
    # function works by creating a 3x3 supercell and only keeping the
    # molecules that don't intersect the boundary of the unit cell
    struc_graph *= (3, 3, 3)
    supercell_g = nx.Graph(struc_graph.graph)

    # Extract all molecule subgraphs
    all_subgraphs = [
        supercell_g.subgraph(c) for c in nx.connected_components(supercell_g)
    ]

    # Only keep that molecules that are completely contained in the 3x3 supercell
    molecule_subgraphs = []
    for subgraph in all_subgraphs:
        intersects_boundary = any(
            d["to_jimage"] != (0, 0, 0)
            for u, v, d in subgraph.edges(data=True)
        )
        if not intersects_boundary:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))

    # Get the center of mass and the molecule index
    center_of_masses = []
    site_props = list(s.site_properties.keys())
    # site_props.remove("molecule_index")
    props = {p: [] for p in site_props}
    for subgraph in molecule_subgraphs:
        cart_coords = np.vstack(
            [struc_graph.structure[n].coords for n in subgraph]
        )
        weights = np.array(
            [struc_graph.structure[n].species.weight for n in subgraph]
        )

        for p in props:
            ind = list(subgraph.nodes.keys())[0]
            props[p].append(struc_graph.structure[ind].properties[p])

        center_of_mass = (
            np.sum(cart_coords * weights[:, None], axis=0) / weights.sum()
        )
        center_of_masses.append(np.round(center_of_mass, 6))

    center_of_masses = np.vstack(center_of_masses)

    # Now we can find which center of masses are contained in the original unit cell
    # First we can shift the center of masses by the [1, 1, 1] vector of the original unit cell
    # so the center unit cell of the 3x3 supercell is positioned at (0, 0, 0)
    shift = s.lattice.get_cartesian_coords([1, 1, 1])
    inv_matrix = s.lattice.inv_matrix

    # Shift the center of masses
    center_of_masses -= shift

    # Convert to fractional coordinates in the basis of the original unit cell
    frac_com = center_of_masses.dot(inv_matrix)

    # The center of masses in the unit cell should have fractional coordinates between [0, 1)
    in_original_cell = np.logical_and(
        0 <= np.round(frac_com, 6), np.round(frac_com, 6) < 1
    ).all(axis=1)

    # Extract the fractional coordinates in the original cell
    frac_coords_in_cell = frac_com[in_original_cell]
    props_in_cell = {
        p: [l[i] for i in np.where(in_original_cell)[0]]
        for p, l in props.items()
    }

    # Extract the molecules who's center of mass is in the original cell
    molecules = []
    for i in np.where(in_original_cell)[0]:
        m_graph = molecule_subgraphs[i]
        coords = [struc_graph.structure[n].coords for n in m_graph.nodes()]
        species = [struc_graph.structure[n].specie for n in m_graph.nodes()]
        molecule = Molecule(species, coords)
        molecule = molecule.get_centered_molecule()
        molecules.append(molecule)

    # Create the structure with the center of mass
    # species, frac_coords, bases, mols = list(zip(*struc_data))
    if "dummy_species" not in props_in_cell:
        species = [i + 22 for i in range(len(molecules))]
        props_in_cell["dummy_species"] = species
    else:
        species = props_in_cell["dummy_species"]

    frac_coords = frac_coords_in_cell
    struc_props = {
        "molecules": molecules,
    }
    struc_props.update(props_in_cell)

    dummy_struc = Structure(
        lattice=s.lattice,
        coords=frac_coords,
        species=species,
        site_properties=struc_props,
    )
    dummy_struc.sort()

    return dummy_struc


def add_molecules(struc):
    mol_coords = []
    mol_atom_nums = []

    properties = list(struc.site_properties.keys())
    properties.remove("molecules")
    site_props = {p: [] for p in properties}
    site_props["molecule_index"] = []

    for i, site in enumerate(struc):
        site_mol = site.properties["molecules"]
        mol_coords.append(site_mol.cart_coords + site.coords)
        mol_atom_nums.extend(site_mol.atomic_numbers)

        site_props["molecule_index"].extend([i] * len(site_mol))

        for p in properties:
            site_props[p].extend([site.properties[p]] * len(site_mol))

    mol_layer_struc = Structure(
        lattice=struc.lattice,
        species=mol_atom_nums,
        coords=np.vstack(mol_coords),
        to_unit_cell=True,
        coords_are_cartesian=True,
        site_properties=site_props,
    )
    mol_layer_struc.sort()

    return mol_layer_struc


def conv_a_to_b(struc_a: Structure, struc_b: Structure) -> np.ndarray:
    return struc_b.lattice.matrix @ struc_a.lattice.inv_matrix


def get_atoms(struc):
    return AseAtomsAdaptor().get_atoms(struc)


def get_layer_supercelll(
    structure: Structure, layers: int, vacuum_scale: int = 0
) -> Structure:
    base_frac_coords = structure.frac_coords
    sc_base_frac_coords = np.vstack(
        [base_frac_coords + np.array([0, 0, i]) for i in range(layers)]
    )
    sc_cart_coords = sc_base_frac_coords.dot(structure.lattice.matrix)
    sc_layer_inds = np.repeat(np.arange(layers), len(structure))

    new_site_properties = {
        k: v * layers for k, v in structure.site_properties.items()
    }
    new_site_properties["layer_index"] = sc_layer_inds.tolist()

    if "atomic_layer_index" in new_site_properties:
        atomic_layers = np.array(new_site_properties["atomic_layer_index"])
        offset = (atomic_layers.max() * sc_layer_inds) + sc_layer_inds
        new_atomic_layers = atomic_layers + offset
        new_site_properties["atomic_layer_index"] = new_atomic_layers

    layer_transform = np.eye(3)
    layer_transform[-1, -1] = layers + vacuum_scale
    layer_matrix = layer_transform @ structure.lattice.matrix

    layer_slab = Structure(
        lattice=Lattice(matrix=layer_matrix),
        species=structure.species * layers,
        coords=sc_cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=new_site_properties,
    )

    return layer_slab


def group_layers(structure, atol=None):
    """
    This function will find the atom indices belonging to each unique atomic layer.

    Args:
        structure (pymatgen.core.structure.Structure): Slab structure
        atol (float or None): Tolarence used for grouping the layers. Useful for grouping
            layers in a structure with relaxed atomic positions.

    Returns:
        A list containing the indices of each layers.
        A list of heights of each layers in fractional coordinates.
    """
    sites = structure.sites
    zvals = np.array([site.c for site in sites])
    unique_values = np.sort(np.unique(np.round(zvals, 3)))
    diff = np.mean(np.diff(unique_values)) * 0.2

    grouped = False
    groups = []
    group_heights = []
    zvals_copy = copy.deepcopy(zvals)
    while not grouped:
        if len(zvals_copy) > 0:
            if atol is None:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
            else:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=atol)
                )[0]

            group_heights.append(np.min(zvals_copy))
            zvals_copy = np.delete(
                zvals_copy,
                np.where(np.isin(zvals_copy, zvals[group_index]))[0],
            )
            groups.append(group_index)
        else:
            grouped = True

    return groups, np.array(group_heights)


def get_reduced_basis(basis: np.ndarray) -> np.ndarray:
    """
    This function is used to find the miller indices of the slab structure
    basis vectors in their most reduced form. i.e.

    |  2  4  0 |     | 1  2  0 |
    |  0 -2  4 | ==> | 0 -1  2 |
    | 10 10 10 |     | 1  1  1 |

    Args:
        basis (np.ndarray): 3x3 matrix defining the lattice vectors

    Returns:
        Reduced integer basis in the form of miller indices
    """
    basis /= np.linalg.norm(basis, axis=1)[:, None]

    for i, b in enumerate(basis):
        basis[i] = _get_reduced_vector(b)

    return basis


def _get_reduced_vector(vector: np.ndarry) -> np.ndarray:
    """ """
    abs_b = np.abs(vector)
    vector /= abs_b[abs_b > 0.001].min()
    vector /= np.abs(reduce(_float_gcd, vector))

    return np.round(vector)


def _float_gcd(a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def reduce_vectors_zur_and_mcgill(a, b):
    vecs = np.vstack([a, b])
    mat = np.eye(3)
    reduced = False

    while not reduced:
        dot = np.round(np.dot(vecs[0], vecs[1]), 6)
        a_norm = np.round(np.linalg.norm(vecs[0]), 6)
        b_norm = np.round(np.linalg.norm(vecs[1]), 6)
        b_plus_a_norm = np.round(np.linalg.norm(vecs[1] + vecs[0]), 6)
        b_minus_a_norm = np.round(np.linalg.norm(vecs[1] - vecs[0]), 6)

        if dot < 0:
            vecs[1] *= -1
            mat[1] *= -1
            continue

        if a_norm > b_norm:
            vecs = vecs[[1, 0]]
            mat = mat[[1, 0, 2]]
            continue

        if b_norm > b_plus_a_norm:
            vecs[1] = vecs[1] + vecs[0]
            mat[1] = mat[1] + mat[0]
            continue

        if b_norm > b_minus_a_norm:
            vecs[1] = vecs[1] - vecs[0]
            mat[1] = mat[1] - mat[0]
            reduced = True
            continue

        reduced = True

    final_dot = np.dot(vecs[0], vecs[1])
    dot_0 = np.isclose(np.round(final_dot, 5), 0.0)
    a_norm = np.linalg.norm(vecs[0])
    b_norm = np.linalg.norm(vecs[1])

    basis = np.eye(3)
    basis[:2] = vecs
    det = np.linalg.det(basis)
    lefty = det < 0

    if dot_0 and lefty:
        vecs[1] *= -1
        mat[1] *= -1

    if not dot_0 and np.isclose(a_norm, b_norm) and lefty:
        vecs = vecs[[1, 0]]
        mat = mat[[1, 0, 2]]

    return vecs[0], vecs[1], mat


def get_primitive_structure(
    struc,
    tolerance: float = 0.25,
    use_site_props: bool = False,
    constrain_latt: list | dict | None = None,
):
    """
    This finds a smaller unit cell than the input. Sometimes it doesn"t
    find the smallest possible one, so this method is recursively called
    until it is unable to find a smaller cell.

    NOTE: if the tolerance is greater than 1/2 the minimum inter-site
    distance in the primitive cell, the algorithm will reject this lattice.

    Args:
        tolerance (float), Angstroms: Tolerance for each coordinate of a
            particular site. For example, [0.1, 0, 0.1] in cartesian
            coordinates will be considered to be on the same coordinates
            as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.
        use_site_props (bool): Whether to account for site properties in
            differentiating sites.
        constrain_latt (list/dict): List of lattice parameters we want to
            preserve, e.g. ["alpha", "c"] or dict with the lattice
            parameter names as keys and values we want the parameters to
            be e.g. {"alpha": 90, "c": 2.5}.

    Returns:
        The most primitive structure found.
    """
    if constrain_latt is None:
        constrain_latt = []

    def site_label(site):
        if not use_site_props:
            return site.species_string
        d = [site.species_string]
        for k in sorted(site.properties.keys()):
            d.append(k + "=" + str(site.properties[k]))
        return ", ".join(d)

    # group sites by species string
    sites = sorted(struc._sites, key=site_label)

    grouped_sites = [
        list(a[1]) for a in itertools.groupby(sites, key=site_label)
    ]
    grouped_fcoords = [
        np.array([s.frac_coords for s in g]) for g in grouped_sites
    ]

    # min_vecs are approximate periodicities of the cell. The exact
    # periodicities from the supercell matrices are checked against these
    # first
    min_fcoords = min(grouped_fcoords, key=lambda x: len(x))
    min_vecs = min_fcoords - min_fcoords[0]

    # fractional tolerance in the supercell
    super_ftol = np.divide(tolerance, struc.lattice.abc)
    super_ftol_2 = super_ftol * 2

    def pbc_coord_intersection(fc1, fc2, tol):
        """
        Returns the fractional coords in fc1 that have coordinates
        within tolerance to some coordinate in fc2
        """
        d = fc1[:, None, :] - fc2[None, :, :]
        d -= np.round(d)
        np.abs(d, d)
        return fc1[np.any(np.all(d < tol, axis=-1), axis=-1)]

    # here we reduce the number of min_vecs by enforcing that every
    # vector in min_vecs approximately maps each site onto a similar site.
    # The subsequent processing is O(fu^3 * min_vecs) = O(n^4) if we do no
    # reduction.
    # This reduction is O(n^3) so usually is an improvement. Using double
    # the tolerance because both vectors are approximate
    for g in sorted(grouped_fcoords, key=lambda x: len(x)):
        for f in g:
            min_vecs = pbc_coord_intersection(min_vecs, g - f, super_ftol_2)

    def get_hnf(fu):
        """
        Returns all possible distinct supercell matrices given a
        number of formula units in the supercell. Batches the matrices
        by the values in the diagonal (for less numpy overhead).
        Computational complexity is O(n^3), and difficult to improve.
        Might be able to do something smart with checking combinations of a
        and b first, though unlikely to reduce to O(n^2).
        """

        def factors(n):
            for i in range(1, n + 1):
                if n % i == 0:
                    yield i

        for det in factors(fu):
            if det == 1:
                continue
            for a in factors(det):
                for e in factors(det // a):
                    g = det // a // e
                    yield det, np.array(
                        [
                            [[a, b, c], [0, e, f], [0, 0, g]]
                            for b, c, f in itertools.product(
                                range(a), range(a), range(e)
                            )
                        ]
                    )

    # we can't let sites match to their neighbors in the supercell
    grouped_non_nbrs = []
    for gfcoords in grouped_fcoords:
        fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
        fdist -= np.round(fdist)
        np.abs(fdist, fdist)
        non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
        # since we want sites to match to themselves
        np.fill_diagonal(non_nbrs, True)
        grouped_non_nbrs.append(non_nbrs)

    num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
    for size, ms in get_hnf(num_fu):
        inv_ms = np.linalg.inv(ms)

        # find sets of lattice vectors that are are present in min_vecs
        dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
        dist -= np.round(dist)
        np.abs(dist, dist)
        is_close = np.all(dist < super_ftol, axis=-1)
        any_close = np.any(is_close, axis=-1)
        inds = np.all(any_close, axis=-1)

        for inv_m, m in zip(inv_ms[inds], ms[inds]):
            new_m = np.dot(inv_m, struc.lattice.matrix)
            ftol = np.divide(tolerance, np.sqrt(np.sum(new_m**2, axis=1)))

            valid = True
            new_coords = []
            new_sp = []
            new_props = collections.defaultdict(list)
            for gsites, gfcoords, non_nbrs in zip(
                grouped_sites, grouped_fcoords, grouped_non_nbrs
            ):
                all_frac = np.dot(gfcoords, m)

                # calculate grouping of equivalent sites, represented by
                # adjacency matrix
                fdist = all_frac[None, :, :] - all_frac[:, None, :]
                fdist = np.abs(fdist - np.round(fdist))
                close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                groups = np.logical_and(close_in_prim, non_nbrs)

                # check that groups are correct
                if not np.all(np.sum(groups, axis=0) == size):
                    valid = False
                    break

                # check that groups are all cliques
                for g in groups:
                    if not np.all(groups[g][:, g]):
                        valid = False
                        break
                if not valid:
                    break

                # add the new sites, averaging positions
                added = np.zeros(len(gsites))
                new_fcoords = all_frac % 1
                for i, group in enumerate(groups):
                    if not added[i]:
                        added[group] = True
                        inds = np.where(group)[0]
                        coords = new_fcoords[inds[0]]
                        for n, j in enumerate(inds[1:]):
                            offset = new_fcoords[j] - coords
                            coords += (offset - np.round(offset)) / (n + 2)
                        new_sp.append(gsites[inds[0]].species)
                        for k in gsites[inds[0]].properties:
                            new_props[k].append(gsites[inds[0]].properties[k])
                        new_coords.append(coords)

            if valid:
                inv_m = np.linalg.inv(m)
                new_l = Lattice(np.dot(inv_m, struc.lattice.matrix))
                s = Structure(
                    new_l,
                    new_sp,
                    new_coords,
                    site_properties=new_props,
                    coords_are_cartesian=False,
                )

                p = get_primitive_structure(
                    s,
                    tolerance=tolerance,
                    use_site_props=use_site_props,
                    constrain_latt=constrain_latt,
                )
                if not constrain_latt:
                    return p

                # Only return primitive structures that
                # satisfy the restriction condition
                p_latt, s_latt = p.lattice, struc.lattice
                if type(constrain_latt).__name__ == "list":
                    if all(
                        getattr(p_latt, pp) == getattr(s_latt, pp)
                        for pp in constrain_latt
                    ):
                        return p
                elif type(constrain_latt).__name__ == "dict":
                    if all(
                        getattr(p_latt, pp) == constrain_latt[pp] for pp in constrain_latt.keys()  # type: ignore
                    ):
                        return p

    return struc.copy()
