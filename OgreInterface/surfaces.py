"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface import utils
from OgreInterface.lattice_match import OgreMatch

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import CrystalNN, BrunnerNN_real

from typing import Dict, Union, Iterable, List, Tuple, TypeVar
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Shadow
from matplotlib.colors import to_rgb, to_rgba
from itertools import combinations, groupby
import numpy as np
import copy
from copy import deepcopy
from functools import reduce
from ase import Atoms
import warnings

# suppress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")

SelfSurface = TypeVar("SelfSurface", bound="Surface")
SelfInterface = TypeVar("SelfInterface", bound="Interface")


class Surface:
    """Container for surfaces generated with the SurfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Examples:
        Generating a surface with pseudo-hydrogen passivation where the atomic positions of the hydrogens need to be relaxed using DFT.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True)
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=True) # relax=True will automatically set selective dynamics=True for all passivating hydrogens

        Generating a surface with pseudo-hydrogen passivation that comes from a structure with pre-relaxed pseudo-hydrogens.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=20, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True, passivated_struc="CONTCAR") # CONTCAR is the output of the structural relaxation
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=False)

    Args:
        orthogonal_slab: Slab structure that is forced to have an c lattice vector that is orthogonal
            to the inplane lattice vectors
        non_orthogonal_slab: Slab structure that is not gaurunteed to have an orthogonal c lattice vector,
            and assumes the same basis as the primitive_oriented_bulk structure.
        oriented_bulk: Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk: Bulk structure that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix: 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index: Miller indices of the surface, with respect to the conventional bulk structure.
        layers: Number of unit cell layers in the surface
        vacuum: Size of the vacuum in Angstroms
        uvw_basis: Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations: List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist: z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist: z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index: Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal

    Attributes:
        oriented_bulk_structure: Pymatgen Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        oriented_bulk_atoms (Atoms): ASE Atoms of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk_structure (Structure): Bulk Pymatgen Structure that can be transformed into the slab basis using the transformation_matrix
        bulk_atoms (Atoms): Bulk ASE Atoms that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix (np.ndarray): 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index (list): Miller indices of the surface, with respect to the conventional bulk structure.
        layers (int): Number of unit cell layers in the surface
        vacuum (float): Size of the vacuum in Angstroms
        uvw_basis (np.ndarray): Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations (np.ndarray): List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index (int): Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        orthogonal_slab: Union[Structure, Atoms],
        non_orthogonal_slab: Union[Structure, Atoms],
        oriented_bulk: Union[Structure, Atoms],
        bulk: Union[Structure, Atoms],
        transformation_matrix: np.ndarray,
        miller_index: list,
        layers: int,
        vacuum: float,
        uvw_basis: np.ndarray,
        point_group_operations: np.ndarray,
        bottom_layer_dist: float,
        top_layer_dist: float,
        termination_index: int,
        surface_normal: np.ndarray,
        c_projection: int,
    ) -> None:
        (
            self._orthogonal_slab_structure,
            self._orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(orthogonal_slab)
        (
            self._non_orthogonal_slab_structure,
            self._non_orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(non_orthogonal_slab)
        (
            self.oriented_bulk_structure,
            self.oriented_bulk_atoms,
        ) = self._get_atoms_and_struc(oriented_bulk)
        (
            self.bulk_structure,
            self.bulk_atoms,
        ) = self._get_atoms_and_struc(bulk)

        self.surface_normal = surface_normal
        self.c_projection = c_projection
        self._transformation_matrix = transformation_matrix
        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.uvw_basis = uvw_basis
        self.point_group_operations = point_group_operations
        self.bottom_layer_dist = bottom_layer_dist
        self.top_layer_dist = top_layer_dist
        self.termination_index = termination_index
        self._passivated = False

    def get_surface(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the surface structure generated from the SurfaceGenerator

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the surface structure
        """

        if orthogonal:
            return_struc = self._orthogonal_slab_structure
        else:
            return_struc = self._non_orthogonal_slab_structure

        if "molecules" in return_struc.site_properties:
            return_struc = utils.add_molecules(return_struc)

        if return_atoms:
            return utils.get_atoms(return_struc)
        else:
            return return_struc

    def get_layer_indices(
        self, layer: int, atomic_layers: bool = True
    ) -> np.ndarray:
        """
        This function is used to extract the atom-indicies of specific layers of the surface.

        Examples:
            >>> surface.get_layer_indices(layer=0)
            >>> [0 1 2 3]

        Args:
            layer: The layer number of the surface which you would like to get atom-indices for.
            atomic_layers: Determines if it is in terms of atomic layers or unit cell layers

        Returns:
            A numpy array of integer indices corresponding to the atom index of the surface structure
        """
        if atomic_layers:
            layer_key = "atomic_layer_index"
        else:
            layer_key = "layer_index"

        surface = self._non_orthogonal_slab_structure
        site_props = surface.site_properties
        layer_index = np.array(site_props[layer_key])
        return np.where(layer_index == layer)[0]

    @property
    def top_surface_charge(self) -> float:
        frac_coords = self.oriented_bulk_structure.frac_coords
        charges = np.array(
            self.oriented_bulk_structure.site_properties["charge"]
        )

        z_frac = 1 - np.mod(np.round(frac_coords[:, -1], 5), 1)

        return np.round((charges * z_frac).sum(), 4)

    @property
    def bottom_surface_charge(self) -> float:
        frac_coords = self.oriented_bulk_structure.frac_coords
        charges = np.array(
            self.oriented_bulk_structure.site_properties["charge"]
        )
        z_frac = np.mod(np.round(frac_coords[:, -1], 5), 1)

        return np.round((charges * z_frac).sum(), 4)

    @property
    def atomic_layers(self) -> int:
        """
        This function will return the number of atomic layers in the slab
        """
        return int(
            max(
                self._non_orthogonal_slab_structure.site_properties[
                    "atomic_layer_index"
                ]
            )
            + 1
        )

    @property
    def slab_transformation_matrix(self) -> np.ndarray:
        """
        Transformation matrix to convert the primitive bulk lattice vectors to the
        slab supercell lattice vectors (including the vacuum region)

        Examples:
            >>> surface.slab_transformation_matrix
            >>> [[ -1   1   0]
            ...  [  0   0   1]
            ...  [ 15  15 -15]]
        """
        layer_mat = np.eye(3)
        layer_mat[-1, -1] = self.layers + np.round(
            self.vacuum / self.c_projection
        )

        return (layer_mat @ self._transformation_matrix).astype(int)

    @property
    def bulk_transformation_matrix(self) -> np.ndarray:
        """
        Transformation matrix to convert the primitive bulk unit cell to the smallest
        oriented unit cell of the slab structure

        Examples:
            >>> surface.bulk_transformation_matrix
            >>> [[ -1   1   0]
            ...  [  0   0   1]
            ...  [  1   1  -1]]
        """
        return self._transformation_matrix.astype(int)

    @property
    def formula(self) -> str:
        """
        Reduced formula of the surface

        Examples:
            >>> surface.formula
            >>> "InAs"

        Returns:
            Reduced formula of the underlying bulk structure
        """
        return self.bulk_structure.composition.reduced_formula

    @property
    def area(self) -> float:
        """
        Cross section area of the slab in Angstroms^2

        Examples:
            >>> surface.area
            >>> 62.51234

        Returns:
            Cross-section area in Angstroms^2
        """
        area = np.linalg.norm(
            np.cross(
                self._orthogonal_slab_structure.lattice.matrix[0],
                self._orthogonal_slab_structure.lattice.matrix[1],
            )
        )

        return area

    @property
    def inplane_vectors(self) -> np.ndarray:
        """
        In-plane cartesian vectors of the slab structure

        Examples:
            >>> surface.inplane_vectors
            >>> [[4.0 0.0 0.0]
            ...  [2.0 2.0 0.0]]

        Returns:
            (2, 3) numpy array containing the cartesian coordinates of the in-place lattice vectors
        """
        matrix = deepcopy(self._orthogonal_slab_structure.lattice.matrix)
        return matrix[:2]

    @property
    def miller_index_a(self) -> np.ndarray:
        """
        Miller index of the a-lattice vector

        Examples:
            >>> surface.miller_index_a
            >>> [-1 1 0]

        Returns:
            (3,) numpy array containing the miller indices
        """
        return self.uvw_basis[0].astype(int)

    @property
    def miller_index_b(self) -> np.ndarray:
        """
        Miller index of the b-lattice vector

        Examples:
            >>> surface.miller_index_b
            >>> [1 -1 0]

        Returns:
            (3,) numpy array containing the miller indices
        """
        return self.uvw_basis[1].astype(int)

    def _get_atoms_and_struc(self, atoms_or_struc) -> Tuple[Structure, Atoms]:
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
            init_atoms = atoms_or_struc
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
            init_atoms = AseAtomsAdaptor.get_atoms(atoms_or_struc)
        else:
            raise TypeError(
                f"Surface._get_atoms_and_struc() accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        return init_structure, init_atoms

    def write_file(
        self,
        output: str = "POSCAR_slab",
        orthogonal: bool = True,
        relax: bool = False,
    ) -> None:
        """
        Writes a POSCAR file of the surface with important information about the slab such as the number of layers, the termination index, and pseudo-hydrogen charges

        Examples:
            Writing a POSCAR file for a static DFT calculation:
            >>> surface.write_file(output="POSCAR", orthogonal=True, relax=False)

            Writing a passivated POSCAR file that needs to be relaxed using DFT:
            >>> surface.write_file(output="POSCAR", orthogonal=True, relax=True)


        Args:
            orthogonal: Determines the the output slab is forced to have a c-vector that is orthogonal to the a and b lattice vectors
            output: File path of the POSCAR
            relax: Determines if selective dynamics should be set in the POSCAR
        """
        if orthogonal:
            slab = self._orthogonal_slab_structure
        else:
            slab = self._non_orthogonal_slab_structure

        if "molecules" in slab.site_properties:
            slab = utils.add_molecules(slab)

        comment = "|".join(
            [
                f"L={self.layers}",
                f"T={self.termination_index}",
                f"O={orthogonal}",
            ]
        )

        if not self._passivated:
            poscar_str = Poscar(slab, comment=comment).get_string()
        else:
            if relax:
                atomic_numbers = np.array(slab.atomic_numbers)
                selective_dynamics = np.repeat(
                    (atomic_numbers == 1).reshape(-1, 1),
                    repeats=3,
                    axis=1,
                )
            else:
                selective_dynamics = None

            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0 and oxi != 0.5:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi == 0.5:
                            H_str = "H.5"
                        elif oxi > 1.0 and oxi != 1.5:
                            H_str = "H" + f"{oxi:.2f}"
                        elif oxi == 1.5:
                            H_str = "H1.5"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = selective_dynamics

            poscar_str = poscar.get_string().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)

    def remove_layers(
        self,
        num_layers: int,
        top: bool = False,
        atol: Union[float, None] = None,
    ) -> None:
        """
        Removes atomic layers from a specified side of the surface. Using this function will ruin the pseudo-hydrogen passivation
        for the side that has layers removed, so it would be prefered to just select a different termination from the list of Surfaces
        generated using the SurfaceGenerator instead of manually removing layers to get the termination you want.

        Examples:
            Removing 3 layers from the top of a surface:
            >>> surface.remove_layers(num_layers=3, top=True)

        Args:
            num_layers: Number of atomic layers to remove
            top: Determines of the layers are removed from the top of the slab or the bottom if False
            atol: Tolarence for grouping the layers, if None, it is automatically determined and usually performs well
        """
        group_inds_conv, _ = utils.group_layers(
            structure=self._orthogonal_slab_structure, atol=atol
        )
        if top:
            group_inds_conv = group_inds_conv[::-1]

        to_delete_conv = []
        for i in range(num_layers):
            to_delete_conv.extend(group_inds_conv[i])

        self._orthogonal_slab_structure.remove_sites(to_delete_conv)

    def _get_surface_atoms(self, cutoff: float) -> Tuple[Structure, List]:
        obs = self.oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_guess()

        layer_struc = utils.get_layer_supercelll(structure=obs, layers=3)
        layer_struc.sort()

        layer_inds = np.array(layer_struc.site_properties["layer_index"])

        bottom_inds = np.where(layer_inds == 0)[0]
        top_inds = np.where(layer_inds == np.max(layer_inds))[0]

        cnn = CrystalNN(search_cutoff=cutoff)
        # cnn = BrunnerNN_real(cutoff=3.0)
        top_neighborhood = []
        for i in top_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] > 0:
                    top_neighborhood.append((i, info_dict))
                    break

        bottom_neighborhood = []
        for i in bottom_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] < 0:
                    bottom_neighborhood.append((i, info_dict))
                    break

        neighborhool_list = [bottom_neighborhood, top_neighborhood]

        return layer_struc, neighborhool_list

    def _get_pseudohydrogen_charge(
        self,
        site,
        coordination,
        include_d_valence: bool = True,
        manual_oxidation_states: Union[Dict[str, float], None] = None,
    ) -> float:
        electronic_struc = site.specie.electronic_structure.split(".")[1:]

        # TODO automate anion/cation determination
        if manual_oxidation_states:
            species_str = str(site.specie._el)
            oxi_state = manual_oxidation_states[species_str]
        else:
            oxi_state = site.specie.oxi_state

        valence = 0
        for orb in electronic_struc:
            if include_d_valence:
                if orb[1] == "d":
                    if int(orb[2:]) < 10:
                        valence += int(orb[2:])
                else:
                    valence += int(orb[2:])
            else:
                if orb[1] != "d":
                    valence += int(orb[2:])

        if oxi_state < 0:
            charge = (8 - valence) / coordination
        else:
            charge = ((2 * coordination) - valence) / coordination

        available_charges = np.array(
            [
                0.25,
                0.33,
                0.42,
                0.5,
                0.58,
                0.66,
                0.75,
                1.00,
                1.25,
                1.33,
                1.50,
                1.66,
                1.75,
            ]
        )

        closest_charge = np.abs(charge - available_charges)
        min_diff = np.isclose(closest_charge, closest_charge.min())
        charge = np.min(available_charges[min_diff])

        return charge

    def _get_bond_dict(
        self,
        cutoff: float,
        include_d_valence: bool,
        manual_oxidation_states,
    ) -> Dict[str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]]:
        image_map = {1: "+", 0: "=", -1: "-"}
        (
            layer_struc,
            surface_neighborhoods,
        ) = self._get_surface_atoms(cutoff)

        labels = ["bottom", "top"]
        bond_dict = {"bottom": {}, "top": {}}
        H_len = 0.31

        for i, neighborhood in enumerate(surface_neighborhoods):
            for surface_atom in neighborhood:
                atom_index = surface_atom[0]
                center_atom_equiv_index = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]

                try:
                    center_len = CovalentRadius.radius[
                        layer_struc[atom_index].specie.symbol
                    ]
                except KeyError:
                    center_len = layer_struc[atom_index].specie.atomic_radius

                oriented_bulk_equivalent = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]
                neighbor_info = surface_atom[1]
                coordination = len(neighbor_info)
                charge = self._get_pseudohydrogen_charge(
                    layer_struc[atom_index],
                    coordination,
                    include_d_valence,
                    manual_oxidation_states,
                )
                broken_atoms = [
                    neighbor
                    for neighbor in neighbor_info
                    if neighbor["image"][-1] != 0
                ]

                bonds = []
                bond_strs = []
                for atom in broken_atoms:
                    broken_site = atom["site"]
                    broken_atom_equiv_index = broken_site.properties[
                        "oriented_bulk_equivalent"
                    ]
                    broken_image = np.array(broken_site.image).astype(int)
                    broken_atom_cart_coords = broken_site.coords
                    center_atom_cart_coords = layer_struc[atom_index].coords
                    bond_vector = (
                        broken_atom_cart_coords - center_atom_cart_coords
                    )
                    norm_vector = bond_vector / np.linalg.norm(bond_vector)
                    H_vector = (H_len + center_len) * norm_vector

                    H_str = ",".join(
                        [
                            str(center_atom_equiv_index),
                            str(broken_atom_equiv_index),
                            "".join([image_map[i] for i in broken_image]),
                            str(i),  # top or bottom bottom=0, top=1
                        ]
                    )

                    bonds.append(H_vector)
                    bond_strs.append(H_str)

                bond_dict[labels[i]][oriented_bulk_equivalent] = {
                    "bonds": np.vstack(bonds),
                    "bond_strings": bond_strs,
                    "charge": charge,
                }

        return bond_dict

    def _get_passivation_atom_index(
        self, struc, bulk_equivalent, top=False
    ) -> int:
        struc_layer_index = np.array(struc.site_properties["layer_index"])
        struc_bulk_equiv = np.array(
            struc.site_properties["oriented_bulk_equivalent"]
        )

        if top:
            layer_number = np.max(struc_layer_index)
        else:
            layer_number = 0

        atom_index = np.where(
            np.logical_and(
                struc_layer_index == layer_number,
                struc_bulk_equiv == bulk_equivalent,
            )
        )[0][0]

        return atom_index

    def _passivate(self, struc, index, bond, bond_str, charge) -> None:
        position = struc[index].coords + bond
        frac_coords = np.mod(
            np.round(struc.lattice.get_fractional_coords(position), 6), 1
        )
        props = {k: -1 for k in struc[index].properties}
        props["hydrogen_str"] = f"{index}," + bond_str

        struc.append(
            Species("H", oxidation_state=charge),
            coords=frac_coords,
            coords_are_cartesian=False,
            properties=props,
        )

    def _get_passivated_bond_dict(
        self,
        bond_dict: Dict[
            str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]
        ],
        relaxed_structure_file: str,
    ) -> Dict[str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]]:
        # Load in the relaxed structure file to get the description string
        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        # Get the description string at the top of the POSCAR/CONTCAR
        desc_str = poscar_str[0].split("|")

        # Extract the number of layers
        layers = int(desc_str[0].split("=")[1])

        # Extract the termination index
        termination_index = int(desc_str[1].split("=")[1])

        # If the termination index is the same the proceed with passivation
        if termination_index == self.termination_index:
            # Extract the structure
            structure = Structure.from_file(relaxed_structure_file)

            # Make a copy of the oriented bulk structure
            obs = self.oriented_bulk_structure.copy()

            # Add oxidation states for the passivation
            obs.add_oxidation_state_by_guess()

            # If the OBS is left handed make it right handed like the pymatgen Poscar class does
            is_negative = np.linalg.det(obs.lattice.matrix) < 0

            if is_negative:
                structure = Structure(
                    lattice=Lattice(structure.lattice.matrix * -1),
                    species=structure.species,
                    coords=structure.frac_coords,
                )

            # Reproduce the passivated structure
            vacuum_scale = 4
            layer_struc = utils.get_layer_supercelll(
                structure=obs, layers=layers, vacuum_scale=vacuum_scale
            )

            center_shift = 0.5 * (vacuum_scale / (vacuum_scale + layers))
            layer_struc.translate_sites(
                indices=range(len(layer_struc)),
                vector=[0, 0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )

            layer_struc.sort()

            # Add hydrogen_str propery. This avoids the PyMatGen warning
            layer_struc.add_site_property(
                "hydrogen_str", [-1] * len(layer_struc)
            )

            # Add a site propery indexing each atom before the passivation is applied
            layer_struc.add_site_property(
                "pre_passivation_index", list(range(len(layer_struc)))
            )

            # Get top and bottom species to determine if the layer_struc should be
            # passivated on the top or bottom of the structure
            atomic_numbers = structure.atomic_numbers
            top_species = atomic_numbers[
                np.argmax(structure.frac_coords[:, -1])
            ]
            bot_species = atomic_numbers[
                np.argmin(structure.frac_coords[:, -1])
            ]

            # If the top species is a Hydrogen then passivate the top
            if top_species == 1:
                for bulk_equiv, bonds in bond_dict["top"].items():
                    ortho_index = self._get_passivation_atom_index(
                        struc=layer_struc, bulk_equivalent=bulk_equiv, top=True
                    )

                    for bond, bond_str in zip(
                        bonds["bonds"], bonds["bond_strings"]
                    ):
                        self._passivate(
                            layer_struc,
                            ortho_index,
                            bond,
                            bond_str,
                            bonds["charge"],
                        )

            # If the bottom species is a Hydrogen then passivate the bottom
            if bot_species == 1:
                for bulk_equiv, bonds in bond_dict["bottom"].items():
                    ortho_index = self._get_passivation_atom_index(
                        struc=layer_struc,
                        bulk_equivalent=bulk_equiv,
                        top=False,
                    )

                    for bond, bond_str in zip(
                        bonds["bonds"], bonds["bond_strings"]
                    ):
                        self._passivate(
                            layer_struc,
                            ortho_index,
                            bond,
                            bond_str,
                            bonds["charge"],
                        )

            layer_struc.sort()

            shifts = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ).dot(structure.lattice.matrix)

            # Get the index if the hydrogens
            hydrogen_index = np.where(np.array(structure.atomic_numbers) == 1)[
                0
            ]

            # Get the bond strings from the passivated structure
            bond_strs = layer_struc.site_properties["hydrogen_str"]

            # Get the index of sites before passivation
            pre_pas_inds = layer_struc.site_properties["pre_passivation_index"]

            # The bond center of the hydrogens are the first element of the bond string
            pre_pas_bond_centers = [
                int(bond_strs[i].split(",")[0]) for i in hydrogen_index
            ]

            # Map the pre-passivation bond index to the actual index in the passivated structure
            post_pas_bond_centers = [
                pre_pas_inds.index(i) for i in pre_pas_bond_centers
            ]

            # Get the coordinates of the bond centers in the actual relaxed structure
            # and the recreated ideal passivated structure
            relaxed_bond_centers = structure.cart_coords[post_pas_bond_centers]
            ideal_bond_centers = layer_struc.cart_coords[post_pas_bond_centers]

            # Get the coordinates of the hydrogens in the actual relaxed structure
            # and the recreated ideal passivated structure
            relaxed_hydrogens = structure.cart_coords[hydrogen_index]
            ideal_hydrogens = layer_struc.cart_coords[hydrogen_index]

            # Substract the bond center positions from the hydrogen positions to get only the bond vector
            relaxed_hydrogens -= relaxed_bond_centers
            ideal_hydrogens -= ideal_bond_centers

            # Mapping to accessing the bond_dict
            top_bot_dict = {1: "top", 0: "bottom"}

            # Lopp through the matching hydrogens and indices to get the difference between the bond vectors
            for H_ind, H_ideal, H_relaxed in zip(
                hydrogen_index, ideal_hydrogens, relaxed_hydrogens
            ):
                # Find all periodic shifts of the relaxed hydrogens
                relaxed_shifts = H_relaxed + shifts

                # Find the difference between the ideal hydrogens and all 3x3 periodic images of the relaxed hydrogen
                diffs = relaxed_shifts - H_ideal

                # Find the length of the bond difference vectors
                norm_diffs = np.linalg.norm(diffs, axis=1)

                # Find the difference vector between the ideal hydrogen and the closest relaxed hydrogen image
                bond_diff = diffs[np.argmin(norm_diffs)]

                # Get the bond string of the hydrogen
                bond_str = bond_strs[H_ind].split(",")

                # Extract the side from the bond string (the last element)
                side = top_bot_dict[int(bond_str[-1])]

                # Get the center index
                center_ind = int(bond_str[1])

                # Extract the bond info from the bond_dict
                bond_info = bond_dict[side][center_ind]

                # Find which bond this hydrogen corresponds to
                bond_ind = bond_info["bond_strings"].index(
                    ",".join(bond_str[1:])
                )

                # Add the bond diff to the bond to get the relaxed position
                bond_dict[side][center_ind]["bonds"][bond_ind] += bond_diff

            return bond_dict
        else:
            raise ValueError(
                f"This is not the same termination. The passivated structure has termination={termination_index}, and the current surface has termination={self.termination_index}"
            )

    def passivate(
        self,
        bottom: bool = True,
        top: bool = True,
        cutoff: float = 4.0,
        passivated_struc: Union[str, None] = None,
        include_d_valence: bool = True,
        manual_oxidation_states: Union[Dict[str, float], None] = None,
    ) -> None:
        """
        This function will apply pseudohydrogen passivation to all broken bonds on the surface and assign charges to the pseudo-hydrogens based
        on the equations provided in https://doi.org/10.1103/PhysRevB.85.195328. The identification of the local coordination environments is
        provided using CrystalNN in Pymatgen which is based on https://doi.org/10.1021/acs.inorgchem.0c02996.

        Examples:
            Initial passivation:
            >>> surface.passivate(bottom=True, top=True)

            Relaxed passivation from a CONTCAR file:
            >>> surface.passivate(bottom=True, top=True, passivated_struc="CONTCAR")

        Args:
            bottom: Determines if the bottom of the structure should be passivated
            top: Determines of the top of the structure should be passivated
            cutoff: Determines the cutoff in Angstroms for the nearest neighbor search. 3.0 seems to give reasonalble reasults.
            passivated_struc: File path to the CONTCAR/POSCAR file that contains the relaxed atomic positions of the pseudo-hydrogens.
                This structure must have the same miller index and termination index.
            include_d_valence: (DO NOT CHANGE FROM DEFAULT, THIS IS ONLY FOR DEBUGING) Determines if the d-orbital electrons are included the calculation of the pseudohydrogen charge.
            manual_oxidation_states:  (DO NOT CHANGE FROM DEFAULT, THIS IS ONLY FOR DEBUGING) Option to pass in a dictionary determining which elements are anions vs cations.
                This will be automated hopefully at some point.
                (i.e {"Ti": 1, "Mn": 1, "In": -1} would mean Ti and Mn are cations and In is an anion)
        """
        bond_dict = self._get_bond_dict(
            cutoff, include_d_valence, manual_oxidation_states
        )

        if passivated_struc is not None:
            bond_dict = self._get_passivated_bond_dict(
                bond_dict=bond_dict, relaxed_structure_file=passivated_struc
            )

        ortho_slab = self._orthogonal_slab_structure.copy()
        non_ortho_slab = self._non_orthogonal_slab_structure.copy()
        ortho_slab.add_site_property("hydrogen_str", [-1] * len(ortho_slab))
        non_ortho_slab.add_site_property(
            "hydrogen_str", [-1] * len(non_ortho_slab)
        )

        if top:
            for bulk_equiv, bonds in bond_dict["top"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        if bottom:
            for bulk_equiv, bonds in bond_dict["bottom"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        ortho_slab.sort()
        non_ortho_slab.sort()

        ortho_slab.remove_site_property("hydrogen_str")
        non_ortho_slab.remove_site_property("hydrogen_str")

        self._passivated = True
        self._orthogonal_slab_structure = ortho_slab
        self._non_orthogonal_slab_structure = non_ortho_slab

    def get_termination(self):
        """
        Returns the termination of the surface as a dictionary

        Examples:
            >>> surface.get_termination()
            >>> {"bottom": {"In": 1, "As": 0}, "top": {"In": 0, "As": 1}
        """
        raise NotImplementedError


class Interface:
    """Container of Interfaces generated using the InterfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Args:
        substrate: Surface class of the substrate material
        film: Surface class of the film material
        match: OgreMatch class of the matching interface
        interfacial_distance: Distance between the top atom of the substrate and the bottom atom of the film
        vacuum: Size of the vacuum in Angstroms
        center: Determines if the interface is centered in the vacuum

    Attributes:
        substrate (Surface): Surface class of the substrate material
        film (Surface): Surface class of the film material
        match (OgreMatch): OgreMatch class of the matching interface
        interfacial_distance (float): Distance between the top atom of the substrate and the bottom atom of the film
        vacuum (float): Size of the vacuum in Angstroms
        center (bool): Determines if the interface is centered in the vacuum
    """

    def __init__(
        self,
        substrate: Union[Surface, SelfInterface],
        film: Union[Surface, SelfInterface],
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
    ) -> None:
        self.center = center
        self.substrate = substrate
        self.film = film
        self.match = match
        self.vacuum = vacuum
        (
            self._substrate_supercell,
            self._substrate_obs_supercell,
            self._substrate_supercell_uvw,
            self._substrate_supercell_scale_factors,
        ) = self._create_supercell(substrate=True)
        (
            self._film_supercell,
            self._film_obs_supercell,
            self._film_supercell_uvw,
            self._film_supercell_scale_factors,
        ) = self._create_supercell(substrate=False)
        self._substrate_a_to_i = self._orient_supercell(
            supercell=self._substrate_supercell
        )
        self._film_a_to_i = self._orient_supercell(
            supercell=self._film_supercell
        )

        self.interfacial_distance = interfacial_distance
        self._strained_sub = self._substrate_supercell
        self._strained_film, self._strain_matrix = self._prepare_film()

        (
            self._M_matrix,
            self._non_orthogonal_structure,
            self._non_orthogonal_substrate_structure,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_substrate_atoms,
            self._non_orthogonal_film_atoms,
            self._orthogonal_structure,
            self._orthogonal_substrate_structure,
            self._orthogonal_film_structure,
            self._orthogonal_atoms,
            self._orthogonal_substrate_atoms,
            self._orthogonal_film_atoms,
        ) = self._stack_interface()
        self._a_shift = 0.0
        self._b_shift = 0.0
        self.bottom_layer_dist = self.substrate.bottom_layer_dist
        self.top_layer_dist = self.film.top_layer_dist

    @property
    def inplane_vectors(self) -> np.ndarray:
        """
        In-plane cartesian vectors of the interface structure

        Examples:
            >>> interface.inplane_vectors
            >>> [[4.0 0.0 0.0]
            ...  [2.0 2.0 0.0]]

        Returns:
            (2, 3) numpy array containing the cartesian coordinates of the in-place lattice vectors
        """
        matrix = deepcopy(self._orthogonal_structure.lattice.matrix)
        return matrix[:2]

    @property
    def uvw_basis(self) -> np.ndarray:
        return np.eye(3).astype(int)

    @property
    def oriented_bulk_structure(self) -> Structure:
        return self.substrate.oriented_bulk_structure

    @property
    def substrate_oriented_bulk_supercell(self) -> Structure:
        if self._substrate_obs_supercell is not None:
            return self._substrate_obs_supercell
        else:
            raise "substrate_oriented_bulk_supercell is not applicable when an Interface is used as the substrate"

    @property
    def film_oriented_bulk_supercell(self) -> Structure:
        if self._film_obs_supercell is not None:
            obs_supercell = utils.apply_strain_matrix(
                structure=self._film_obs_supercell,
                strain_matrix=self._strain_matrix,
            )
            return obs_supercell
        else:
            raise "substrate_oriented_bulk_supercell is not applicable when an Interface is used as the substrate"

    @property
    def c_projection(self) -> float:
        return self.substrate.c_projection

    def _passivated(self) -> bool:
        return self.substrate._passivated or self.film._passivated

    @property
    def _transformation_matrix(self) -> np.ndarray:
        return self.substrate._transformation_matrix

    @property
    def surface_normal(self) -> np.ndarray:
        return self.substrate.surface_normal

    @property
    def layers(self) -> int:
        return self.substrate.layers + self.film.layers

    @property
    def atomic_layers(self) -> int:
        return self.substrate.atomic_layers + self.film.atomic_layers

    @property
    def termination_index(self) -> int:
        return 0

    @property
    def point_group_operations(self) -> np.ndarray:
        sg = SpacegroupAnalyzer(self._orthogonal_structure)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

    @property
    def transformation_matrix(self):
        """
        Transformation matrix to convert the primitive bulk lattice vectors of the substrate material to the
        interface supercell lattice vectors (including the vacuum region)

        Examples:
            >>> interface.transformation_matrix
            >>> [[ -2   2   0]
            ...  [  0   0   2]
            ...  [ 15  15 -15]]
        """
        return self._M_matrix.astype(int)

    def get_interface(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the interface structure generated from the OgreMatch

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the interface structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_atoms
            else:
                return self._orthogonal_structure
        else:
            if return_atoms:
                return self._non_orthogonal_atoms
            else:
                return self._non_orthogonal_structure

    def get_substrate_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the substrate supercell generated from the OgreMatch
        (i.e. the interface structure with the film atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the substrate supercell structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_substrate_atoms
            else:
                return self._orthogonal_substrate_structure
        else:
            if return_atoms:
                return self._non_orthogonal_substrate_atoms
            else:
                return self._non_orthogonal_substrate_structure

    def get_film_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the film supercell generated from the OgreMatch
        (i.e. the interface structure with the substrate atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the film supercell structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_film_atoms
            else:
                return self._orthogonal_film_structure
        else:
            if return_atoms:
                return self._non_orthogonal_film_atoms
            else:
                return self._non_orthogonal_film_structure

    def get_substrate_layer_indices(
        self,
        layer_from_interface: int,
        atomic_layers: bool = True,
    ) -> np.ndarray:
        """
        This function is used to extract the atom-indicies of specific layers of the substrate part of the interface.

        Examples:
            >>> interface.get_substrate_layer_indices(layer_from_interface=0)
            >>> [234 235 236 237 254 255 256 257]


        Args:
            layer_from_interface: The layer number of the substrate which you would like to get
                atom-indices for. The layer number is reference from the interface, so layer_from_interface=0
                would be the layer of the substrate that is at the interface.

        Returns:
            A numpy array of integer indices corresponding to the atom index of the interface structure
        """
        if atomic_layers:
            layer_key = "atomic_layer_index"
        else:
            layer_key = "layer_index"

        interface = self._non_orthogonal_structure
        site_props = interface.site_properties
        is_sub = np.array(site_props["is_sub"])
        layer_index = np.array(site_props[layer_key])
        sub_n_layers = layer_index[is_sub].max()
        rel_layer_index = sub_n_layers - layer_index
        is_layer = rel_layer_index == layer_from_interface

        return np.where(np.logical_and(is_sub, is_layer))[0]

    def get_film_layer_indices(
        self,
        layer_from_interface: int,
        atomic_layers: bool = True,
    ) -> np.ndarray:
        """
        This function is used to extract the atom-indicies of specific layers of the film part of the interface.

        Examples:
            >>> interface.get_substrate_layer_indices(layer_from_interface=0)
            >>> [0 1 2 3 4 5 6 7 8 9 10 11 12]

        Args:
            layer_from_interface: The layer number of the film which you would like to get atom-indices for.
            The layer number is reference from the interface, so layer_from_interface=0
            would be the layer of the film that is at the interface.

        Returns:
            A numpy array of integer indices corresponding to the atom index of the interface structure
        """
        if atomic_layers:
            layer_key = "atomic_layer_index"
        else:
            layer_key = "layer_index"

        interface = self._non_orthogonal_structure
        site_props = interface.site_properties
        is_film = np.array(site_props["is_film"])
        layer_index = np.array(site_props[layer_key])
        is_layer = layer_index == layer_from_interface

        return np.where(np.logical_and(is_film, is_layer))[0]

    def replace_species(
        self, site_index: int, species_mapping: Dict[str, str]
    ) -> None:
        """
        This function can be used to replace the species at a given site in the interface structure

        Examples:
            >>> interface.replace_species(site_index=42, species_mapping={"In": "Zn", "As": "Te"})

        Args:
            site_index: Index of the site to be replaced
            species_mapping: Dictionary showing the mapping between species.
                For example if you wanted to replace InAs with ZnTe then the species mapping would
                be as shown in the example above.
        """
        species_str = self._orthogonal_structure[site_index].species_string

        if species_str in species_mapping:
            is_sub = self._non_orthogonal_structure[site_index].properties[
                "is_sub"
            ]
            self._non_orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )
            self._orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )

            if is_sub:
                sub_iface_equiv = np.array(
                    self._orthogonal_substrate_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                sub_site_ind = np.where(sub_iface_equiv == site_index)[0][0]
                self._non_orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
            else:
                film_iface_equiv = np.array(
                    self._orthogonal_film_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                film_site_ind = np.where(film_iface_equiv == site_index)[0][0]
                self._non_orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
        else:
            raise ValueError(
                f"Species: {species_str} is not is species mapping"
            )

    @property
    def area(self) -> float:
        """
        Cross section area of the interface in Angstroms^2

        Examples:
            >>> interface.area
            >>> 205.123456

        Returns:
            Cross-section area in Angstroms^2
        """
        return self.match.area

    @property
    def _structure_volume(self) -> float:
        matrix = deepcopy(self._orthogonal_structure.lattice.matrix)
        vac_matrix = np.vstack(
            [
                matrix[:2],
                self.vacuum * (matrix[-1] / np.linalg.norm(matrix[-1])),
            ]
        )

        total_volume = np.abs(np.linalg.det(matrix))
        vacuum_volume = np.abs(np.linalg.det(vac_matrix))

        return total_volume - vacuum_volume

    @property
    def substrate_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the substrate supercell

        Examples:
            >>> interface.substrate_basis
            >>> [[3 1 0]
            ...  [-1 3 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._substrate_supercell_uvw

    @property
    def substrate_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_a
            >>> [3 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._substrate_supercell_uvw[0]

    @property
    def substrate_b(self) -> np.ndarray:
        """
        Returns the miller indices of the b basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_b
            >>> [-1 3 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._substrate_supercell_uvw[1]

    @property
    def film_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the film supercell

        Examples:
            >>> interface.film_basis
            >>> [[1 -1 0]
            ...  [0 1 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._film_supercell_uvw

    @property
    def film_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_a
            >>> [1 -1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._film_supercell_uvw[0]

    @property
    def film_b(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_b
            >>> [0 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._film_supercell_uvw[1]

    def __str__(self):
        fm = self.film.miller_index
        sm = self.substrate.miller_index
        film_str = f"{self.film.formula}({fm[0]} {fm[1]} {fm[2]})"
        sub_str = f"{self.substrate.formula}({sm[0]} {sm[1]} {sm[2]})"
        s_uvw = self._substrate_supercell_uvw
        s_sf = self._substrate_supercell_scale_factors
        f_uvw = self._film_supercell_uvw
        f_sf = self._film_supercell_scale_factors
        match_a_film = (
            f"{f_sf[0]}*[{f_uvw[0][0]:2d} {f_uvw[0][1]:2d} {f_uvw[0][2]:2d}]"
        )
        match_a_sub = (
            f"{s_sf[0]}*[{s_uvw[0][0]:2d} {s_uvw[0][1]:2d} {s_uvw[0][2]:2d}]"
        )
        match_b_film = (
            f"{f_sf[1]}*[{f_uvw[1][0]:2d} {f_uvw[1][1]:2d} {f_uvw[1][2]:2d}]"
        )
        match_b_sub = (
            f"{s_sf[1]}*[{s_uvw[1][0]:2d} {s_uvw[1][1]:2d} {s_uvw[1][2]:2d}]"
        )
        return_info = [
            "Film: " + film_str,
            "Substrate: " + sub_str,
            "Epitaxial Match Along \\vec{a} (film || sub): "
            + f"({match_a_film} || {match_a_sub})",
            "Epitaxial Match Along \\vec{b} (film || sub): "
            + f"({match_b_film} || {match_b_sub})",
            "Strain Along \\vec{a} (%): "
            + f"{100*self.match.linear_strain[0]:.3f}",
            "Strain Along \\vec{b} (%): "
            + f"{100*self.match.linear_strain[1]:.3f}",
            "In-plane Angle Mismatch (%): "
            + f"{100*self.match.angle_strain:.3f}",
            "Cross Section Area (Ang^2): " + f"{self.area:.3f}",
        ]
        return_str = "\n".join(return_info)

        return return_str

    def _load_relaxed_structure(
        self, relaxed_structure_file: str
    ) -> np.ndarray:

        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        desc_str = poscar_str[0].split("|")

        layers = desc_str[0].split("=")[1].split(",")
        termination_index = desc_str[1].split("=")[1].split(",")
        ortho = bool(int(desc_str[2].split("=")[1]))
        d_int = desc_str[3].split("=")[1]
        layers_to_relax = desc_str[4].split("=")[1].split(",")

        film_layers = int(layers[0])
        sub_layers = int(layers[1])

        film_termination_index = int(termination_index[0])
        sub_termination_index = int(termination_index[1])

        N_film_layers_to_relax = int(layers_to_relax[0])
        N_sub_layers_to_relax = int(layers_to_relax[1])

        if (
            d_int == f"{self.interfacial_distance:.3f}"
            and film_termination_index == self.film.termination_index
            and sub_termination_index == self.substrate.termination_index
        ):
            poscar = Poscar.from_string(
                "\n".join(poscar_str), read_velocities=False
            )
            relaxed_structure = poscar.structure

            if ortho:
                unrelaxed_structure = self._orthogonal_structure.copy()
            else:
                unrelaxed_structure = self._non_orthogonal_structure.copy()

            unrelaxed_structure.add_site_property(
                "orig_ind", list(range(len(unrelaxed_structure)))
            )

            unrelaxed_hydrogen_inds = np.where(
                np.array(unrelaxed_structure.atomic_numbers) == 1
            )[0]

            unrelaxed_structure.remove_sites(unrelaxed_hydrogen_inds)

            is_negative = np.linalg.det(unrelaxed_structure.lattice.matrix) < 0

            if is_negative:
                relaxed_structure = Structure(
                    lattice=Lattice(relaxed_structure.lattice.matrix * -1),
                    species=relaxed_structure.species,
                    coords=relaxed_structure.frac_coords,
                )

            hydrogen_inds = np.where(
                np.array(relaxed_structure.atomic_numbers) == 1
            )[0]

            relaxed_structure.remove_sites(hydrogen_inds)

            relaxation_shifts = np.zeros((len(unrelaxed_structure), 3))

            is_film_full = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_full = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_full = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )
            sub_to_delete = np.logical_and(
                is_sub_full,
                layer_index_full < self.substrate.layers - sub_layers,
            )

            film_to_delete = np.logical_and(
                is_film_full, layer_index_full >= film_layers
            )

            to_delete = np.where(np.logical_or(sub_to_delete, film_to_delete))[
                0
            ]

            unrelaxed_structure.remove_sites(to_delete)

            is_film_small = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_small = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_small = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )

            film_layers_to_relax = np.arange(N_film_layers_to_relax)

            sub_layers_to_relax = np.arange(
                self.substrate.layers - N_sub_layers_to_relax,
                self.substrate.layers,
            )

            film_to_relax = np.logical_and(
                is_film_small, np.isin(layer_index_small, film_layers_to_relax)
            )
            sub_to_relax = np.logical_and(
                is_sub_small, np.isin(layer_index_small, sub_layers_to_relax)
            )

            relaxed_inds = np.where(
                np.logical_or(film_to_relax, sub_to_relax)
            )[0]

            periodic_shifts = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ).dot(unrelaxed_structure.lattice.matrix)

            ref_ind = np.min(np.where(is_sub_small)[0])
            unrelaxed_ref = unrelaxed_structure[ref_ind].coords
            relaxed_ref = relaxed_structure[ref_ind].coords

            for i in relaxed_inds:
                init_ind = unrelaxed_structure[i].properties["orig_ind"]
                relaxed_coords = relaxed_structure[i].coords
                relaxed_coords -= relaxed_ref
                unrelaxed_coords = unrelaxed_structure[i].coords
                unrelaxed_coords -= unrelaxed_ref

                all_relaxed_coords = periodic_shifts + relaxed_coords
                dists = np.linalg.norm(
                    all_relaxed_coords - unrelaxed_coords, axis=1
                )
                center_ind = np.argmin(dists)
                bond = all_relaxed_coords[center_ind] - unrelaxed_coords
                relaxation_shifts[init_ind] = bond

            return relaxation_shifts
        else:
            raise ValueError(
                "The surface terminations and interfacial distances must be the same"
            )

    def relax_interface(self, relaxed_structure_file: str) -> None:
        """
        This function will shift the positions of the atoms near the interface coorresponding to the
        atomic positions from a relaxed interface. This especially usefull when running DFT on large interface
        structures because the atomic positions can be relaxed using an interface with less layers, and
        then the relax positions can be applied to a much larger interface for a static DFT calculation.

        Examples:
            >>> interface.relax_interface(relax_structure_file="CONTCAR")

        Args:
            relaxed_structure_file: File path to the relax structure (CONTCAR/POSCAR for now)
        """
        relaxation_shifts = self._load_relaxed_structure(
            relaxed_structure_file
        )
        init_ortho_structure = self._orthogonal_structure
        non_hydrogen_inds = np.array(init_ortho_structure.atomic_numbers) != 1
        new_coords = init_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_ortho_structure = Structure(
            lattice=init_ortho_structure.lattice,
            species=init_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_ortho_structure.site_properties,
        )
        relaxed_ortho_atoms = utils.get_atoms(relaxed_ortho_structure)
        (
            relaxed_ortho_film_structure,
            relaxed_ortho_film_atoms,
            relaxed_ortho_sub_structure,
            relaxed_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(relaxed_ortho_structure)

        init_non_ortho_structure = self._non_orthogonal_structure
        non_hydrogen_inds = (
            np.array(init_non_ortho_structure.atomic_numbers) != 1
        )
        new_coords = init_non_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_non_ortho_structure = Structure(
            lattice=init_non_ortho_structure.lattice,
            species=init_non_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_non_ortho_structure.site_properties,
        )
        relaxed_non_ortho_atoms = utils.get_atoms(relaxed_non_ortho_structure)
        (
            relaxed_non_ortho_film_structure,
            relaxed_non_ortho_film_atoms,
            relaxed_non_ortho_sub_structure,
            relaxed_non_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(relaxed_non_ortho_structure)

        self._orthogonal_structure = relaxed_ortho_structure
        self._orthogonal_film_structure = relaxed_ortho_film_structure
        self._orthogonal_substrate_structure = relaxed_ortho_sub_structure
        self._non_orthogonal_structure = relaxed_non_ortho_structure
        self._non_orthogonal_film_structure = relaxed_non_ortho_film_structure
        self._non_orthogonal_substrate_structure = (
            relaxed_non_ortho_sub_structure
        )
        self._orthogonal_atoms = relaxed_ortho_atoms
        self._orthogonal_film_atoms = relaxed_ortho_film_atoms
        self._orthogonal_substrate_atoms = relaxed_ortho_sub_atoms
        self._non_orthogonal_atoms = relaxed_non_ortho_atoms
        self._non_orthogonal_film_atoms = relaxed_non_ortho_film_atoms
        self._non_orthogonal_substrate_atoms = relaxed_non_ortho_sub_atoms

    def write_file(
        self,
        output: str = "POSCAR_interface",
        orthogonal: bool = True,
        relax: bool = False,
        film_layers_to_relax: int = 1,
        substrate_layers_to_relax: int = 1,
        atomic_layers: bool = False,
        relax_z_only: bool = False,
    ) -> None:
        """
        Write the POSCAR of the interface

        Args:
            output: File path of the output POSCAR
            orthogonal: Determines of the orthogonal structure is written
            relax: Determines if selective dynamics is applied to the atoms at the interface
            film_layers_to_relax: Number of unit cell layers near the interface to relax
            substrate_layers_to_relax: Number of unit cell layers near the interface to relax
            atomic_layers: Determines if the number of layer is atomic layers or unit cell layers
            relax_z_only: Determines if the relaxation is in the z-direction only
        """
        if orthogonal:
            slab = self._orthogonal_structure
        else:
            slab = self._non_orthogonal_structure

        comment = "|".join(
            [
                f"L={self.film.layers},{self.substrate.layers}",
                f"T={self.film.termination_index},{self.substrate.termination_index}",
                f"O={int(orthogonal)}",
                f"d={self.interfacial_distance:.3f}",
            ]
        )

        if relax:
            comment += "|" + "|".join(
                [
                    f"R={film_layers_to_relax},{substrate_layers_to_relax}",
                ]
            )
            film_layers = np.arange(film_layers_to_relax)

            if atomic_layers:
                layer_key = "atomic_layer_index"
                sub_layers = np.arange(
                    self.substrate.atomic_layers - substrate_layers_to_relax,
                    self.substrate.atomic_layers,
                )
            else:
                layer_key = "layer_index"
                sub_layers = np.arange(
                    self.substrate.layers - substrate_layers_to_relax,
                    self.substrate.layers,
                )

            layer_index = np.array(slab.site_properties[layer_key])
            is_sub = np.array(slab.site_properties["is_sub"])
            is_film = np.array(slab.site_properties["is_film"])
            film_to_relax = np.logical_and(
                is_film, np.isin(layer_index, film_layers)
            )
            sub_to_relax = np.logical_and(
                is_sub, np.isin(layer_index, sub_layers)
            )

            to_relax = np.repeat(
                np.logical_or(sub_to_relax, film_to_relax).reshape(-1, 1),
                repeats=3,
                axis=1,
            )

            if relax_z_only:
                to_relax[:, :2] = False

        comment += "|" + "|".join(
            [
                f"a={self._a_shift:.4f}",
                f"b={self._b_shift:.4f}",
            ]
        )

        if not self.substrate._passivated and not self.film._passivated:
            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string()

        else:
            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0 and oxi != 0.5:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi == 0.5:
                            H_str = "H.5"
                        elif oxi > 1.0 and oxi != 1.5:
                            H_str = "H" + f"{oxi:.2f}"
                        elif oxi == 1.5:
                            H_str = "H1.5"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)

    def _shift_film(
        self, interface: Structure, shift: Iterable, fractional: bool
    ) -> Tuple[Structure, Atoms, Structure, Atoms]:
        shifted_interface_structure = interface.copy()
        film_ind = np.where(
            shifted_interface_structure.site_properties["is_film"]
        )[0]
        shifted_interface_structure.translate_sites(
            indices=film_ind,
            vector=shift,
            frac_coords=fractional,
            to_unit_cell=True,
        )
        shifted_interface_atoms = utils.get_atoms(shifted_interface_structure)
        (
            shifted_film_structure,
            shifted_film_atoms,
            _,
            _,
        ) = self._get_film_and_substrate_parts(shifted_interface_structure)

        return (
            shifted_interface_structure,
            shifted_interface_atoms,
            shifted_film_structure,
            shifted_film_atoms,
        )

    def set_interfacial_distance(self, interfacial_distance: float) -> None:
        """
        Sets a new interfacial distance for the interface by shifting the film in the z-direction

        Examples:
            >>> interface.set_interfacial_distance(interfacial_distance=2.0)

        Args:
            interfacial_distance: New interfacial distance for the interface
        """
        shift = np.array(
            [0.0, 0.0, interfacial_distance - self.interfacial_distance]
        )
        self.interfacial_distance = interfacial_distance
        (
            self._orthogonal_structure,
            self._orthogonal_atoms,
            self._orthogonal_film_structure,
            self._orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=shift,
            fractional=False,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._non_orthogonal_structure,
            shift=shift,
            fractional=False,
        )

    def shift_film_inplane(
        self,
        x_shift: float,
        y_shift: float,
        fractional: bool = False,
    ) -> None:
        """
        Shifts the film in-place over the substrate within the plane of the interface by a given shift vector.

        Examples:
            Shift using fractional coordinates:
            >>> interface.shift_film(shift=[0.5, 0.25], fractional=True)

            Shift using cartesian coordinates:
            >>> interface.shift_film(shift=[4.5, 0.0], fractional=False)

        Args:
            x_shift: Shift in the x or a-vector directions depending on if fractional=True
            y_shift: Shift in the y or b-vector directions depending on if fractional=True
            fractional: Determines if the shift is defined in fractional coordinates
        """
        shift_array = np.array([x_shift, y_shift, 0.0])

        if fractional:
            frac_shift = shift_array
        else:
            frac_shift = (
                self._orthogonal_structure.lattice.get_fractional_coords(
                    shift_array
                )
            )

        self._a_shift += shift_array[0]
        self._b_shift += shift_array[1]

        (
            self._orthogonal_structure,
            self._orthogonal_atoms,
            self._orthogonal_film_structure,
            self._orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._non_orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )

    def _create_supercell(
        self, substrate: bool = True
    ) -> Tuple[Structure, Structure, np.ndarray, np.ndarray]:
        if substrate:
            matrix = self.match.substrate_sl_transform

            if type(self.substrate) == Surface:
                supercell = (
                    self.substrate._non_orthogonal_slab_structure.copy()
                )
                obs_supercell = self.substrate.oriented_bulk_structure.copy()
            elif type(self.substrate) == Interface:
                supercell = self.substrate._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    not_hydrogen = layer_index != -1
                    is_film = supercell.site_properties["is_film"]
                    is_sub = supercell.site_properties["is_sub"]
                    layer_index[(is_film & not_hydrogen)] += (
                        layer_index[is_sub].max() + 1
                    )
                    supercell.add_site_property(layer_key, layer_index)

            basis = self.substrate.uvw_basis
        else:
            matrix = self.match.film_sl_transform

            if type(self.film) == Surface:
                supercell = self.film._non_orthogonal_slab_structure.copy()
                obs_supercell = self.film.oriented_bulk_structure.copy()
            elif type(self.film) == Interface:
                supercell = self.film._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    is_film = supercell.site_properties["is_film"]
                    is_sub = supercell.site_properties["is_sub"]
                    layer_index[is_film] += layer_index[is_sub].max() + 1
                    supercell.add_site_property(layer_key, layer_index)

            basis = self.film.uvw_basis

        supercell.make_supercell(scaling_matrix=matrix)

        if obs_supercell is not None:
            obs_supercell.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell, obs_supercell, uvw_supercell, scale_factors

    def _orient_supercell(self, supercell: Structure) -> np.ndarray:
        matrix = deepcopy(supercell.lattice.matrix)

        a_norm = matrix[0] / np.linalg.norm(matrix[0])
        a_to_i = np.array(
            [[a_norm[0], -a_norm[1], 0], [a_norm[1], a_norm[0], 0], [0, 0, 1]]
        ).T

        a_to_i_operation = SymmOp.from_rotation_and_translation(
            a_to_i, translation_vec=np.zeros(3)
        )

        if "molecules" in supercell.site_properties:
            utils.apply_op_to_mols(supercell, a_to_i_operation)

        supercell.apply_operation(a_to_i_operation)

        return a_to_i

    def _prepare_substrate(self) -> Tuple[Structure, np.ndarray, np.ndarray]:
        matrix = self.match.substrate_sl_transform
        supercell_slab = self.substrate._non_orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.substrate.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _prepare_film(self) -> Structure:
        supercell_slab = self._film_supercell
        sc_matrix = supercell_slab.lattice.matrix
        sub_sc_matrix = self._substrate_supercell.lattice.matrix

        strained_matrix = np.vstack(
            [
                sub_sc_matrix[:2],
                sc_matrix[-1],
            ]
        )

        init_volume = supercell_slab.volume
        strain_volume = np.abs(np.linalg.det(strained_matrix))
        scale_factor = init_volume / strain_volume

        # Maintain constant volume
        strained_matrix[-1] *= scale_factor
        strain_matrix = np.linalg.inv(sc_matrix) @ strained_matrix

        strained_film = Structure(
            lattice=Lattice(strained_matrix),
            species=supercell_slab.species,
            coords=supercell_slab.frac_coords,
            coords_are_cartesian=False,
            to_unit_cell=True,
            site_properties=supercell_slab.site_properties,
        )

        sub_non_orth_c_vec = self._substrate_supercell.lattice.matrix[-1]
        sub_non_orth_c_norm = sub_non_orth_c_vec / np.linalg.norm(
            sub_non_orth_c_vec
        )

        norm = self.film.surface_normal
        proj = np.dot(norm, sub_non_orth_c_norm)
        scale = strained_film.lattice.c / proj

        new_c_matrix = np.vstack(
            [sub_sc_matrix[:2], sub_non_orth_c_norm * scale]
        )

        oriented_film = Structure(
            lattice=Lattice(new_c_matrix),
            species=strained_film.species,
            coords=strained_film.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=strained_film.site_properties,
        )

        return oriented_film, strain_matrix

    # def _prepare_film_old(self) -> Structure:
    #     supercell_slab = self._film_supercell
    #     sc_matrix = supercell_slab.lattice.matrix
    #     sub_sc_matrix = self._substrate_supercell.lattice.matrix

    #     sub_non_orth_c_vec = self._substrate_supercell.lattice.matrix[-1]
    #     sub_non_orth_c_norm = sub_non_orth_c_vec / np.linalg.norm(
    #         sub_non_orth_c_vec
    #     )

    #     norm = self.film.surface_normal
    #     proj = np.dot(norm, sub_non_orth_c_norm)
    #     scale = supercell_slab.lattice.c / proj

    #     new_c_matrix = np.vstack([sc_matrix[:2], sub_non_orth_c_norm * scale])

    #     oriented_film = Structure(
    #         lattice=Lattice(new_c_matrix),
    #         species=supercell_slab.species,
    #         coords=supercell_slab.cart_coords,
    #         coords_are_cartesian=True,
    #         to_unit_cell=True,
    #         site_properties=supercell_slab.site_properties,
    #     )

    #     strained_matrix = np.vstack(
    #         [
    #             sub_sc_matrix[:2],
    #             oriented_film.lattice.matrix[-1],
    #         ]
    #     )

    #     init_volume = oriented_film.volume
    #     strain_volume = np.abs(np.linalg.det(strained_matrix))
    #     scale_factor = init_volume / strain_volume

    #     # Maintain constant volume
    #     strained_matrix[-1] *= scale_factor

    #     strained_film = Structure(
    #         lattice=Lattice(strained_matrix),
    #         species=oriented_film.species,
    #         coords=oriented_film.frac_coords,
    #         coords_are_cartesian=False,
    #         to_unit_cell=True,
    #         site_properties=supercell_slab.site_properties,
    #     )

    #     return strained_film

    # def _prepare_film_old(self) -> Tuple[Structure, np.ndarray, np.ndarray]:
    #     matrix = self.match.film_sl_transform
    #     supercell_slab = self.film._non_orthogonal_slab_structure.copy()
    #     supercell_slab.make_supercell(scaling_matrix=matrix)

    #     sc_matrix = supercell_slab.lattice.matrix
    #     sub_non_orth_c_vec = (
    #         self.substrate._non_orthogonal_slab_structure.lattice.matrix[-1]
    #     )
    #     sub_non_orth_c_norm = sub_non_orth_c_vec / np.linalg.norm(
    #         sub_non_orth_c_vec
    #     )

    #     norm = self.film.surface_normal
    #     proj = np.dot(norm, sub_non_orth_c_norm)
    #     scale = self.film._orthogonal_slab_structure.lattice.c / proj

    #     new_matrix = np.vstack([sc_matrix[:2], sub_non_orth_c_norm * scale])

    #     oriented_supercell_slab = Structure(
    #         lattice=Lattice(new_matrix),
    #         species=supercell_slab.species,
    #         coords=supercell_slab.cart_coords,
    #         coords_are_cartesian=True,
    #         to_unit_cell=True,
    #         site_properties=supercell_slab.site_properties,
    #     )

    #     uvw_supercell = matrix @ self.film.uvw_basis
    #     scale_factors = []
    #     for i, b in enumerate(uvw_supercell):
    #         scale = np.abs(reduce(utils._float_gcd, b))
    #         uvw_supercell[i] = uvw_supercell[i] / scale
    #         scale_factors.append(scale)

    #     return oriented_supercell_slab, uvw_supercell, scale_factors

    # def _strain_and_orient_film(self) -> Tuple[Structure, np.ndarray]:
    #     sub_in_plane_vecs = self._substrate_supercell.lattice.matrix[:2]
    #     film_out_of_plane = self._film_supercell.lattice.matrix[-1]
    # film_inv_matrix = self._film_supercell.lattice.inv_matrix
    # new_matrix = np.vstack([sub_in_plane_vecs, film_out_of_plane])
    # transform = (film_inv_matrix @ new_matrix).T
    #     print(transform)
    # op = SymmOp.from_rotation_and_translation(
    #     transform, translation_vec=np.zeros(3)
    # )

    # strained_film = deepcopy(self._film_supercell)
    # strained_film.apply_operation(op)

    #     return strained_film, transform

    # def _strain_and_orient_film(self) -> Tuple[Structure, np.ndarray]:
    #     sub_matrix = deepcopy(self._substrate_supercell.lattice.matrix)
    #     film_matrix = deepcopy(self._film_supercell.lattice.matrix)

    #     sub_a = sub_matrix[0] / np.linalg.norm(sub_matrix[0])
    #     film_a = film_matrix[0] / np.linalg.norm(film_matrix[0])

    #     normal = self.film.surface_normal

    #     sub_b = np.cross(sub_a, normal)
    #     sub_b /= np.linalg.norm(sub_b)

    #     film_b = np.cross(film_a, normal)
    #     film_b /= np.linalg.norm(film_b)

    #     sub_ortho_basis = np.vstack(
    #         [
    #             sub_a,
    #             sub_b,
    #             normal,
    #         ]
    #     )

    #     film_ortho_basis = np.vstack(
    #         [
    #             film_a,
    #             film_b,
    #             normal,
    #         ]
    #     )

    #     rotation = (np.linalg.inv(film_ortho_basis) @ sub_ortho_basis).T

    #     op = SymmOp.from_rotation_and_translation(
    #         rotation, translation_vec=np.zeros(3)
    #     )

    #     rotated_film = deepcopy(self._film_supercell)
    #     rotated_film.apply_operation(op)

    #     old_matrix = np.vstack([film_matrix[:2], normal])
    #     new_matrix = np.vstack([sub_matrix[:2], normal])
    #     transform = (np.linalg.inv(old_matrix) @ new_matrix).T

    #     # Poscar(self._film_supercell).write_file("POSCAR_film_init")

    #     # strained_film = Structure(
    #     #     lattice=Lattice(new_matrix),
    #     #     species=self._film_supercell.species,
    #     #     coords=self._film_supercell.frac_coords,
    #     #     to_unit_cell=True,
    #     #     coords_are_cartesian=False,
    #     #     site_properties=self._film_supercell.site_properties,
    #     # )
    #     # Poscar(strained_film).write_file("POSCAR_str_film")

    #     return (
    #         rotated_film,
    #         transform,
    #     )

    def _stack_interface(
        self,
    ) -> Tuple[
        np.ndarray,
        Structure,
        Structure,
        Structure,
        Atoms,
        Atoms,
        Atoms,
        Structure,
        Structure,
        Structure,
        Atoms,
        Atoms,
        Atoms,
    ]:

        # Get the strained substrate and film
        strained_sub = self._strained_sub
        strained_film = self._strained_film

        if "molecules" in strained_sub.site_properties:
            strained_sub = utils.add_molecules(strained_sub)

        if "molecules" in strained_film.site_properties:
            strained_film = utils.add_molecules(strained_film)

        # Get the oriented bulk structure of the substrate
        oriented_bulk_c = self.substrate.oriented_bulk_structure.lattice.c

        # Get the normalized projection of the substrate c-vector onto the normal vector,
        # This is used to determine the length of the non-orthogonal c-vector in order to get
        # the correct vacuum size.
        c_norm_proj = self.substrate.c_projection / oriented_bulk_c

        # Get the substrate matrix and c-vector
        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        # Get the fractional and cartesian coordinates of the substrate and film
        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        # Find the min and max coordinates of the substrate and film
        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        # Get the lengths of the c-vetors of the substrate and film
        sub_c_len = strained_sub.lattice.c
        film_c_len = strained_film.lattice.c

        # Find the total length of the interface structure including the interfacial distance
        interface_structure_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.interfacial_distance / c_norm_proj,
            ]
        )

        # Find the length of the vacuum region in the non-orthogonal basis
        interface_vacuum_len = self.vacuum / c_norm_proj

        # The total length of the interface c-vector should be the length of the structure + length of the vacuum
        # This will get changed in the next line to be exactly an integer multiple of the
        # oriented bulk cell of the substrate
        init_interface_c_len = interface_structure_len + interface_vacuum_len

        # Find the closest integer multiple of the substrate oriented bulk c-vector length
        n_unit_cell = int(np.ceil(init_interface_c_len / oriented_bulk_c))

        # Make the new interface c-vector length an integer multiple of the oriented bulk c-vector
        interface_c_len = oriented_bulk_c * n_unit_cell

        # Create the transformation matrix from the primtive bulk structure to the interface unit cell
        # this is only needed for band unfolding purposes
        sub_M = self.substrate._transformation_matrix
        layer_M = np.eye(3).astype(int)
        layer_M[-1, -1] = n_unit_cell
        interface_M = layer_M @ self.match.substrate_sl_transform @ sub_M

        # Create the new interface lattice vectors
        interface_matrix = np.vstack(
            [sub_matrix[:2], interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)

        # Convert the interfacial distance into fractional coordinated because they are easier to work with
        frac_int_distance_shift = np.array(
            [0, 0, self.interfacial_distance]
        ).dot(interface_lattice.inv_matrix)

        interface_inv_matrix = interface_lattice.inv_matrix

        # Convert the substrate cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to zero
        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        # Convert the film cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to the top substrate c-position + interfacial distance
        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += sub_interface_coords[:, -1].max()
        film_interface_coords += frac_int_distance_shift

        # Combine the coodinates, species, and site_properties to make the interface Structure
        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
            if key in strained_sub.site_properties
            and key in strained_film.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        # Create the non-orthogonalized interface structure
        non_ortho_interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        non_ortho_interface_struc.sort()

        non_ortho_interface_struc.add_site_property(
            "interface_equivalent", list(range(len(non_ortho_interface_struc)))
        )

        if self.center:
            # Get the new vacuum length, needed for shifting
            vacuum_len = interface_c_len - interface_structure_len

            # Find the fractional coordinates of shifting the structure up by half the amount of vacuum cells
            center_shift = np.ceil(vacuum_len / oriented_bulk_c) // 2
            center_shift *= oriented_bulk_c / interface_c_len

            # Center the structure in the vacuum
            non_ortho_interface_struc.translate_sites(
                indices=range(len(non_ortho_interface_struc)),
                vector=[0.0, 0.0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )

        # Get the frac coords of the non-orthogonalized interface
        frac_coords = non_ortho_interface_struc.frac_coords

        # Find the max c-coord of the substrate
        # This is used to shift the x-y positions of the interface structure so the top atom of the substrate
        # is located at x=0, y=0. This will have no effect of the properties of the interface since all the
        # atoms are shifted, it is more of a visual thing to make the interfaces look nice.
        is_sub = np.array(non_ortho_interface_struc.site_properties["is_sub"])
        sub_frac_coords = frac_coords[is_sub]
        max_c = np.max(sub_frac_coords[:, -1])

        # Find the xy-shift in cartesian coordinates
        cart_shift = np.array([0.0, 0.0, max_c]).dot(
            non_ortho_interface_struc.lattice.matrix
        )
        cart_shift[-1] = 0.0

        # Get the projection of the non-orthogonal c-vector onto the surface normal
        proj_c = np.dot(
            self.substrate.surface_normal,
            non_ortho_interface_struc.lattice.matrix[-1],
        )

        # Get the orthogonalized c-vector of the interface (this conserves the vacuum, but breaks symmetries)
        ortho_c = self.substrate.surface_normal * proj_c

        # Create the orthogonalized lattice vectors
        new_matrix = np.vstack(
            [
                non_ortho_interface_struc.lattice.matrix[:2],
                ortho_c,
            ]
        )

        # Create the orthogonalized structure
        ortho_interface_struc = Structure(
            lattice=Lattice(new_matrix),
            species=non_ortho_interface_struc.species,
            coords=non_ortho_interface_struc.cart_coords,
            site_properties=non_ortho_interface_struc.site_properties,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        # Shift the structure so the top substrate atom's x and y postions are zero, similar to the non-orthogonalized structure
        ortho_interface_struc.translate_sites(
            indices=range(len(ortho_interface_struc)),
            vector=-cart_shift,
            frac_coords=False,
            to_unit_cell=True,
        )

        # The next step is used extract on the film and substrate portions of the interface
        # These can be used for charge transfer calculation
        (
            ortho_film_structure,
            ortho_film_atoms,
            ortho_sub_structure,
            ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(ortho_interface_struc)
        (
            non_ortho_film_structure,
            non_ortho_film_atoms,
            non_ortho_sub_structure,
            non_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(non_ortho_interface_struc)

        non_ortho_interface_atoms = utils.get_atoms(non_ortho_interface_struc)
        ortho_interface_atoms = utils.get_atoms(ortho_interface_struc)

        return (
            interface_M,
            non_ortho_interface_struc,
            non_ortho_sub_structure,
            non_ortho_film_structure,
            non_ortho_interface_atoms,
            non_ortho_sub_atoms,
            non_ortho_film_atoms,
            ortho_interface_struc,
            ortho_sub_structure,
            ortho_film_structure,
            ortho_interface_atoms,
            ortho_sub_atoms,
            ortho_film_atoms,
        )

    def _get_film_and_substrate_parts(
        self, interface: Structure
    ) -> Tuple[Structure, Atoms, Structure, Atoms]:
        film_inds = np.where(interface.site_properties["is_film"])[0]
        sub_inds = np.where(interface.site_properties["is_sub"])[0]

        film_structure = interface.copy()
        film_structure.remove_sites(sub_inds)
        film_atoms = utils.get_atoms(film_structure)

        sub_structure = interface.copy()
        sub_structure.remove_sites(film_inds)
        sub_atoms = utils.get_atoms(sub_structure)

        return film_structure, film_atoms, sub_structure, sub_atoms

    @property
    def _metallic_elements(self):
        elements_list = np.array(
            [
                "Li",
                "Be",
                "Na",
                "Mg",
                "Al",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Cs",
                "Ba",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Rn",
                "Fr",
                "Ra",
                "Ac",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds ",
                "Rg ",
                "Cn ",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
            ]
        )
        return elements_list

    def _get_radii(self):
        sub_species = np.unique(
            np.array(self.substrate.bulk_structure.species, dtype=str)
        )
        film_species = np.unique(
            np.array(self.film.bulk_structure.species, dtype=str)
        )

        sub_elements = [Element(s) for s in sub_species]
        film_elements = [Element(f) for f in film_species]

        sub_metal = np.isin(sub_species, self._metallic_elements)
        film_metal = np.isin(film_species, self._metallic_elements)

        if sub_metal.all():
            sub_dict = {
                sub_species[i]: sub_elements[i].metallic_radius
                for i in range(len(sub_elements))
            }
        else:
            Xs = [e.X for e in sub_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                sub_dict = {
                    sub_species[i]: sub_elements[i].average_ionic_radius
                    for i in range(len(sub_elements))
                }
            else:
                sub_dict = {s: CovalentRadius.radius[s] for s in sub_species}

        if film_metal.all():
            film_dict = {
                film_species[i]: film_elements[i].metallic_radius
                for i in range(len(film_elements))
            }
        else:
            Xs = [e.X for e in film_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                film_dict = {
                    film_species[i]: film_elements[i].average_ionic_radius
                    for i in range(len(film_elements))
                }
            else:
                film_dict = {f: CovalentRadius.radius[f] for f in film_species}

        sub_dict.update(film_dict)

        return sub_dict

    def _generate_sc_for_interface_view(
        self, struc, transformation_matrix
    ) -> Tuple[Structure, np.ndarray]:
        plot_struc = Structure(
            lattice=struc.lattice,
            species=["H"],
            coords=np.zeros((1, 3)),
            to_unit_cell=True,
            coords_are_cartesian=True,
        )
        plot_struc.make_supercell(transformation_matrix)
        inv_matrix = plot_struc.lattice.inv_matrix

        return plot_struc, inv_matrix

    def _plot_interface_view(
        self,
        ax,
        zero_coord,
        supercell_shift,
        cell_vetices,
        slab_matrix,
        sc_inv_matrix,
        facecolor,
        edgecolor,
        is_film=False,
        strain=True,
    ) -> None:
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        if is_film:
            if strain:
                strain_matrix = self._strain_matrix
                strain_matrix[-1] = np.array([0, 0, 1])
                plot_coords = cart_coords @ strain_matrix
            else:
                plot_coords = cart_coords

            linewidth = 1.0
        else:
            plot_coords = cart_coords
            linewidth = 2.0

        center = np.round(
            np.mean(cart_coords[:-1], axis=0).dot(sc_inv_matrix),
            3,
        )
        center_in = np.logical_and(-0.0001 <= center[:2], center[:2] <= 1.0001)

        x_in = np.logical_and(fc[:, 0] > 0.0, fc[:, 0] < 1.0)
        y_in = np.logical_and(fc[:, 1] > 0.0, fc[:, 1] < 1.0)
        point_in = np.logical_and(x_in, y_in)

        if point_in.any() or center_in.all():
            poly = Polygon(
                xy=plot_coords[:, :2],
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax.add_patch(poly)

    def _get_image(
        self,
        zero_coord,
        supercell_shift,
        cell_vetices,
        slab_matrix,
        slab_inv_matrix,
        sc_inv_matrix,
    ) -> Union[None, np.ndarray]:
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        center = np.round(
            np.mean(cart_coords[:-1], axis=0).dot(sc_inv_matrix),
            3,
        )
        center_in = np.logical_and(-0.0001 <= center[:2], center[:2] <= 1.0001)

        x_in = np.logical_and(fc[:, 0] > 0.0, fc[:, 0] < 1.0)
        y_in = np.logical_and(fc[:, 1] > 0.0, fc[:, 1] < 1.0)
        point_in = np.logical_and(x_in, y_in)

        if point_in.any() or center_in.all():
            shifted_zero_coord = zero_coord + supercell_shift
            shifted_zero_frac = shifted_zero_coord.dot(slab_inv_matrix)

            return np.round(shifted_zero_frac).astype(int)
        else:
            return None

    def _get_oriented_cell_and_images(
        self, strain: bool = True
    ) -> List[np.ndarray]:
        sub_struc = self.substrate._orthogonal_slab_structure.copy()
        sub_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self._substrate_a_to_i, translation_vec=np.zeros(3)
        )
        sub_struc.apply_operation(sub_a_to_i_op)

        film_struc = self.film._orthogonal_slab_structure.copy()
        film_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self._film_a_to_i, translation_vec=np.zeros(3)
        )
        film_struc.apply_operation(film_a_to_i_op)

        if strain:
            film_struc = utils.apply_strain_matrix(
                structure=film_struc,
                strain_matrix=self._strain_matrix,
            )

        sub_matrix = sub_struc.lattice.matrix
        film_matrix = film_struc.lattice.matrix

        prim_sub_inv_matrix = sub_struc.lattice.inv_matrix
        prim_film_inv_matrix = film_struc.lattice.inv_matrix

        sub_sc_matrix = deepcopy(self._substrate_supercell.lattice.matrix)
        film_sc_matrix = deepcopy(self._film_supercell.lattice.matrix)

        coords = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        sc_shifts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0],
            ]
        )

        sub_sc_shifts = sc_shifts.dot(sub_sc_matrix)
        film_sc_shifts = sc_shifts.dot(film_sc_matrix)

        sub_struc, sub_inv_matrix = self._generate_sc_for_interface_view(
            struc=sub_struc,
            transformation_matrix=self.match.substrate_sl_transform,
        )

        film_struc, film_inv_matrix = self._generate_sc_for_interface_view(
            struc=film_struc,
            transformation_matrix=self.match.film_sl_transform,
        )

        sub_images = []

        for c in sub_struc.cart_coords:
            for shift in sub_sc_shifts:
                sub_image = self._get_image(
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=sub_matrix,
                    slab_inv_matrix=prim_sub_inv_matrix,
                    sc_inv_matrix=sub_inv_matrix,
                )

                if sub_image is not None:
                    sub_images.append(sub_image)

        film_images = []

        for c in film_struc.cart_coords:
            for shift in film_sc_shifts:
                film_image = self._get_image(
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=film_matrix,
                    slab_inv_matrix=prim_film_inv_matrix,
                    sc_inv_matrix=film_inv_matrix,
                )
                if film_image is not None:
                    film_images.append(film_image)

        return sub_matrix, sub_images, film_matrix, film_images

    # def plot_interface_bu(
    #     self,
    #     output: str = "interface_view.png",
    #     strain: bool = True,
    #     dpi: int = 400,
    #     show_in_colab: bool = False,
    #     # film_color: Union[str, list] = [0, 110 / 255, 144 / 255],
    #     # substrate_color: Union[str, list] = [241 / 255, 143 / 255, 1 / 255],
    #     film_color: Union[str, list] = "firebrick",
    #     substrate_color: Union[str, list] = "blue",
    #     film_alpha: float = 0.0,
    #     substrate_alpha: float = 0.0,
    #     film_linewidth: float = 2,
    #     substrate_linewidth: float = 2,
    # ) -> None:
    #     """
    #     This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

    #     Args:
    #         output: File path for the output image
    #         strain: Determines if the film lattice should be strained so it shows perfectly aligned lattice coincidence sites,
    #             or if the film lattice should be unstrained, giving a better visual of the lattice mismatch.
    #         dpi: dpi (dots per inch) of the output image.
    #             Setting dpi=100 gives reasonably sized images when viewed in colab notebook
    #         show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
    #             if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
    #     """
    #     if type(film_color) == str:
    #         film_rgb = to_rgb(film_color)
    #     else:
    #         film_rgb = tuple(film_color)

    #     if type(substrate_color) == str:
    #         sub_rgb = to_rgb(substrate_color)
    #     else:
    #         sub_rgb = tuple(substrate_color)

    #     (
    #         sub_matrix,
    #         sub_images,
    #         film_matrix,
    #         film_images,
    #     ) = self._get_oriented_cell_and_images(strain=strain)

    #     interface_matrix = self._orthogonal_structure.lattice.matrix

    #     coords = np.array(
    #         [
    #             [0, 0, 0],
    #             [1, 0, 0],
    #             [1, 1, 0],
    #             [0, 1, 0],
    #             [0, 0, 0],
    #         ]
    #     )

    #     interface_coords = coords.dot(interface_matrix)

    #     xlim = [0, 1]
    #     ylim = [0, 1]

    #     a = interface_matrix[0, :2]
    #     b = interface_matrix[1, :2]
    #     borders = np.vstack(
    #         [
    #             xlim[0] * a + ylim[0] * b,
    #             xlim[1] * a + ylim[0] * b,
    #             xlim[1] * a + ylim[1] * b,
    #             xlim[0] * a + ylim[1] * b,
    #             xlim[0] * a + ylim[0] * b,
    #         ]
    #     )
    #     x_size = borders[:, 0].max() - borders[:, 0].min()
    #     y_size = borders[:, 1].max() - borders[:, 1].min()
    #     ratio = y_size / x_size

    #     if ratio < 1:
    #         figx = 5 / ratio
    #         figy = 5
    #     else:
    #         figx = 5
    #         figy = 5 * ratio

    #     fig, ax = plt.subplots(
    #         figsize=(figx, figy),
    #         dpi=dpi,
    #     )

    #     for image in sub_images:
    #         sub_coords = (coords + image).dot(sub_matrix)
    #         poly = Polygon(
    #             xy=sub_coords[:, :2],
    #             closed=True,
    #             facecolor=sub_rgb + (substrate_alpha,),
    #             edgecolor=sub_rgb,
    #             linewidth=substrate_linewidth,
    #             zorder=0,
    #         )
    #         ax.add_patch(poly)

    #     for image in film_images:
    #         film_coords = (coords + image).dot(film_matrix)
    #         poly = Polygon(
    #             xy=film_coords[:, :2],
    #             closed=True,
    #             facecolor=film_rgb + (film_alpha,),
    #             edgecolor=film_rgb,
    #             linewidth=film_linewidth,
    #             zorder=10,
    #         )
    #         ax.add_patch(poly)

    #     grid_x = np.linspace(0, 1, 11)
    #     grid_y = np.linspace(0, 1, 11)

    #     X, Y = np.meshgrid(grid_x, grid_y)

    #     iface_inv_matrix = self._orthogonal_structure.lattice.inv_matrix

    #     frac_shifts = (
    #         np.c_[X.ravel(), Y.ravel(), np.zeros(Y.shape).ravel()]
    #         + film_images[0]
    #     )
    #     cart_shifts = frac_shifts.dot(film_matrix)
    #     # frac_shifts = cart_shifts.dot(iface_inv_matrix)

    #     ax.scatter(
    #         cart_shifts[:, 0],
    #         cart_shifts[:, 1],
    #         c="black",
    #     )

    #     ax.plot(
    #         interface_coords[:, 0],
    #         interface_coords[:, 1],
    #         color="black",
    #         linewidth=4,
    #         zorder=20,
    #     )

    #     ax.set_xlim(interface_coords[:, 0].min(), interface_coords[:, 0].max())
    #     ax.set_ylim(interface_coords[:, 1].min(), interface_coords[:, 1].max())

    #     ax.set_aspect("equal")
    #     ax.axis("off")

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")

    #     if not show_in_colab:
    #         plt.close()

    def _add_arrows(
        self, ax, matrix, arrow_color, color, size, labels, fontsize, linewidth
    ):
        norm_matrix = matrix / np.linalg.norm(matrix, axis=1)[:, None]
        circ = Circle(
            xy=[0, 0],
            radius=1.5,
            edgecolor=arrow_color,
            facecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(circ)

        for i in range(2):
            t = ax.text(
                1.5 * norm_matrix[i, 0],
                1.5 * norm_matrix[i, 1],
                "$" + labels[i] + "$",
                fontsize=fontsize,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", fc="w", ec="black", alpha=1),
            )
            bb = t.get_bbox_patch()
            ax.annotate(
                "",
                xytext=1.5 * norm_matrix[i, :2],
                xy=(0, 0),
                fontsize=fontsize,
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="<-",
                    color=arrow_color,
                    shrinkA=5,
                    shrinkB=0,
                    patchA=bb,
                    patchB=None,
                    connectionstyle="arc3,rad=0",
                    linewidth=linewidth,
                ),
            )

            circ1 = Circle(
                xy=[0, 0],
                radius=0.075,
                edgecolor=arrow_color,
                facecolor=arrow_color,
            )
            ax.add_patch(circ1)

    def _add_sc_labels(
        self, ax, label, vector, fontsize, rotation, linewidth, height
    ):
        if rotation > 0:
            theta = np.deg2rad(rotation)
            rot = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )
            shift = rot.dot(np.array([0, height]))
        else:
            shift = np.array([0, -height])

        ax.text(
            (0.5 * vector[0]) + shift[0],
            (0.5 * vector[1]) + shift[1],
            "$" + label + "$",
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", fc="w", ec="w", alpha=1),
            zorder=101,
            rotation=rotation,
        )
        ax.annotate(
            "",
            xytext=vector[:2] + shift,
            xy=shift,
            fontsize=fontsize,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="<-",
                color="black",
                shrinkA=0,
                shrinkB=0,
                patchA=None,
                patchB=None,
                connectionstyle="arc3,rad=0",
                linewidth=linewidth,
            ),
            zorder=100,
        )

        return shift

    def _setup_label_axis(self, ax):
        ax.set_aspect("equal")
        ax.tick_params(
            left=False, labelleft=False, bottom=False, labelbottom=False
        )
        ax.axis("off")

    def _get_miller_label(self, miller_index):
        label = []
        for i in miller_index:
            if i < 0:
                label.append("\\overline{" + f"{abs(int(i))}" + "}")
            else:
                label.append(f"{int(i)}")

        return "[" + " ".join(label) + "]"

    # def plot_interface_old2(
    #     self,
    #     output: str = "interface_view.png",
    #     strain: bool = True,
    #     dpi: int = 400,
    #     show_in_colab: bool = False,
    #     film_color: Union[str, list] = "firebrick",
    #     substrate_color: Union[str, list] = "blue",
    #     film_alpha: float = 0.05,
    #     substrate_alpha: float = 0.05,
    #     film_linewidth: float = 2,
    #     substrate_linewidth: float = 2,
    # ) -> None:
    #     """
    #     This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

    #     Args:
    #         output: File path for the output image
    #         strain: Determines if the film lattice should be strained so it shows perfectly aligned lattice coincidence sites,
    #             or if the film lattice should be unstrained, giving a better visual of the lattice mismatch.
    #         dpi: dpi (dots per inch) of the output image.
    #             Setting dpi=100 gives reasonably sized images when viewed in colab notebook
    #         show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
    #             if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
    #     """
    #     if type(film_color) == str:
    #         film_rgb = to_rgb(film_color)
    #     else:
    #         film_rgb = tuple(film_color)

    #     if type(substrate_color) == str:
    #         sub_rgb = to_rgb(substrate_color)
    #     else:
    #         sub_rgb = tuple(substrate_color)

    #     (
    #         sub_matrix,
    #         sub_images,
    #         film_matrix,
    #         film_images,
    #     ) = self._get_oriented_cell_and_images(strain=strain)

    #     interface_matrix = self._orthogonal_structure.lattice.matrix

    #     coords = np.array(
    #         [
    #             [0, 0, 0],
    #             [1, 0, 0],
    #             [1, 1, 0],
    #             [0, 1, 0],
    #             [0, 0, 0],
    #         ]
    #     )

    #     interface_coords = coords.dot(interface_matrix)

    #     xlim = [0, 1]
    #     ylim = [0, 1]

    #     a = interface_matrix[0, :2]
    #     b = interface_matrix[1, :2]
    #     borders = np.vstack(
    #         [
    #             xlim[0] * a + ylim[0] * b,
    #             xlim[1] * a + ylim[0] * b,
    #             xlim[1] * a + ylim[1] * b,
    #             xlim[0] * a + ylim[1] * b,
    #             xlim[0] * a + ylim[0] * b,
    #         ]
    #     )
    #     x_size = borders[:, 0].max() - borders[:, 0].min()
    #     y_size = borders[:, 1].max() - borders[:, 1].min()
    #     ratio = y_size / x_size

    #     if ratio < 1:
    #         figx = 5 / ratio
    #         figy = 5
    #     else:
    #         figx = 5
    #         figy = 5 * ratio

    #     # TODO add labels to the longer side, left or bottom/top to make the figure more square
    #     if figy >= figx:
    #         mosaic = """
    #             ACD
    #             BCE
    #         """
    #         x_expand = 2 * (figy / 2.5)
    #         xtot = figx + x_expand

    #         fig_area = xtot * figy

    #         fig = plt.figure(figsize=(xtot, figy))
    #         axd = fig.subplot_mosaic(
    #             mosaic,
    #             gridspec_kw={
    #                 "width_ratios": [
    #                     0.5 * x_expand / xtot,
    #                     figx / xtot,
    #                     0.5 * x_expand / xtot,
    #                 ]
    #             },
    #         )
    #     else:
    #         mosaic = """
    #             AB
    #             CC
    #             DE
    #         """
    #         y_expand = 2 * (figx / 2.5)
    #         ytot = figy + y_expand

    #         fig_area = figx * ytot

    #         fig = plt.figure(figsize=(figx, ytot))
    #         axd = fig.subplot_mosaic(
    #             mosaic,
    #             gridspec_kw={
    #                 "height_ratios": [
    #                     0.5 * y_expand / ytot,
    #                     figy / ytot,
    #                     0.5 * y_expand / ytot,
    #                 ]
    #             },
    #         )

    #     fontsize_scale = 14 / 5.92
    #     fontsize = fontsize_scale * np.sqrt(fig_area)

    #     ax1 = axd["A"]
    #     ax2 = axd["B"]
    #     ax3 = axd["D"]
    #     ax4 = axd["E"]

    #     self._setup_label_axis(ax1)
    #     self._setup_label_axis(ax2)
    #     self._setup_label_axis(ax3)
    #     self._setup_label_axis(ax4)

    #     ax = axd["C"]

    #     sub_a_label = self._get_miller_label(self.substrate.miller_index_a)
    #     sub_b_label = self._get_miller_label(self.substrate.miller_index_b)

    #     film_a_label = self._get_miller_label(self.film.miller_index_a)
    #     film_b_label = self._get_miller_label(self.film.miller_index_b)

    #     sub_sc_a_label = self._get_miller_label(self.substrate_a)
    #     sub_sc_b_label = self._get_miller_label(self.substrate_b)

    #     film_sc_a_label = self._get_miller_label(self.film_a)
    #     film_sc_b_label = self._get_miller_label(self.film_b)

    #     self._add_arrows(
    #         ax=ax1,
    #         matrix=sub_matrix,
    #         arrow_color=sub_rgb,
    #         color=sub_rgb + (0.05,),
    #         size=0.03,
    #         labels=[sub_a_label, sub_b_label],
    #         fontsize=fontsize,
    #     )

    #     self._add_arrows(
    #         ax=ax2,
    #         matrix=film_matrix,
    #         arrow_color=film_rgb,
    #         color=film_rgb + (0.05,),
    #         size=0.03,
    #         labels=[film_a_label, film_b_label],
    #         fontsize=fontsize,
    #     )

    #     self._add_arrows(
    #         ax=ax3,
    #         matrix=interface_matrix,
    #         arrow_color="black",
    #         color=sub_rgb + (0.05,),
    #         size=0.03,
    #         labels=[
    #             f"{int(self._substrate_supercell_scale_factors[0])} \\,"
    #             + sub_sc_a_label,
    #             f"{int(self._substrate_supercell_scale_factors[1])} \\,"
    #             + sub_sc_b_label,
    #         ],
    #         fontsize=fontsize,
    #     )

    #     self._add_arrows(
    #         ax=ax4,
    #         matrix=interface_matrix,
    #         arrow_color="black",
    #         color=film_rgb + (0.05,),
    #         size=0.03,
    #         labels=[
    #             f"{int(self._film_supercell_scale_factors[0])} \\,"
    #             + film_sc_a_label,
    #             f"{int(self._film_supercell_scale_factors[1])} \\,"
    #             + film_sc_b_label,
    #         ],
    #         fontsize=fontsize,
    #     )

    #     for image in sub_images:
    #         sub_coords = (coords + image).dot(sub_matrix)
    #         poly = Polygon(
    #             xy=sub_coords[:, :2],
    #             closed=True,
    #             facecolor=sub_rgb + (substrate_alpha,),
    #             edgecolor=sub_rgb,
    #             linewidth=substrate_linewidth,
    #             zorder=0,
    #         )
    #         ax.add_patch(poly)

    #     for image in film_images:
    #         film_coords = (coords + image).dot(film_matrix)
    #         poly = Polygon(
    #             xy=film_coords[:, :2],
    #             closed=True,
    #             facecolor=film_rgb + (film_alpha,),
    #             edgecolor=film_rgb,
    #             linewidth=film_linewidth,
    #             zorder=10,
    #         )
    #         ax.add_patch(poly)

    #     grid_x = np.linspace(0, 1, 11)
    #     grid_y = np.linspace(0, 1, 11)

    #     X, Y = np.meshgrid(grid_x, grid_y)

    #     sc_shifts = np.array(
    #         [
    #             [1, 0, 0],
    #             [0, 1, 0],
    #             [-1, 0, 0],
    #             [0, -1, 0],
    #             [1, 1, 0],
    #             [-1, -1, 0],
    #             [1, -1, 0],
    #             [-1, 1, 0],
    #         ]
    #     )

    #     for shift in sc_shifts:
    #         shift_coords = (coords + shift).dot(interface_matrix)
    #         poly = Polygon(
    #             xy=shift_coords[:, :2],
    #             closed=True,
    #             facecolor="white",
    #             edgecolor="white",
    #             linewidth=1,
    #             zorder=15,
    #         )
    #         ax.add_patch(poly)

    #     ax.plot(
    #         interface_coords[:, 0],
    #         interface_coords[:, 1],
    #         color="black",
    #         linewidth=3,
    #         zorder=20,
    #     )

    #     x_range = interface_coords[:, 0].max() - interface_coords[:, 0].min()
    #     y_range = interface_coords[:, 1].max() - interface_coords[:, 1].min()
    #     x_margin = 0.05 * x_range
    #     y_margin = 0.05 * y_range

    #     ax.set_xlim(
    #         interface_coords[:, 0].min() - x_margin,
    #         interface_coords[:, 0].max() + x_margin,
    #     )
    #     ax.set_ylim(
    #         interface_coords[:, 1].min() - y_margin,
    #         interface_coords[:, 1].max() + y_margin,
    #     )

    #     ax.set_aspect("equal")
    #     ax.axis("off")

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")

    #     if not show_in_colab:
    #         plt.close()

    def plot_interface(
        self,
        output: str = "interface_view.png",
        strain: bool = True,
        dpi: int = 400,
        film_color: Union[str, list] = "firebrick",
        substrate_color: Union[str, list] = "blue",
        film_alpha: float = 0.05,
        substrate_alpha: float = 0.05,
        film_linewidth: float = 2,
        substrate_linewidth: float = 2,
        show_in_colab: bool = False,
    ) -> None:
        """
        This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

        Args:
            output: File path for the output image
            strain: Determines if the film lattice should be strained so it shows perfectly aligned lattice coincidence sites,
                or if the film lattice should be unstrained, giving a better visual of the lattice mismatch.
            dpi: dpi (dots per inch) of the output image.
                Setting dpi=100 gives reasonably sized images when viewed in colab notebook
            film_color: Color to represent the film lattice vectors
            substrate_color: Color to represent the substrate lattice vectors
            film_alpha: Tranparency of the film color (ranging from 0 to 1)
            substrate_alpha: Tranparency of the substrate color (ranging from 0 to 1)
            film_linewidth: Linewidth of the film lattice vector edges
            substrate_linewidth: Linewidth of the substrate lattice vector edges
            show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
                if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
        """
        if type(film_color) == str:
            film_rgb = to_rgb(film_color)
        else:
            film_rgb = tuple(film_color)

        if type(substrate_color) == str:
            sub_rgb = to_rgb(substrate_color)
        else:
            sub_rgb = tuple(substrate_color)

        (
            sub_matrix,
            sub_images,
            film_matrix,
            film_images,
        ) = self._get_oriented_cell_and_images(strain=strain)

        interface_matrix = self._orthogonal_structure.lattice.matrix

        coords = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        interface_coords = coords.dot(interface_matrix)

        a = interface_matrix[0, :2]
        b = interface_matrix[1, :2]

        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()
        ratio = y_size / x_size

        if ratio < 1:
            figx = 6 / ratio
            figy = 6
        else:
            figx = 6
            figy = 6 * ratio

        # TODO add labels to the longer side, left or bottom/top to make the figure more square
        scale = 2.75
        if figy >= figx:
            mosaic = """
                AC
                BC
            """
            x_expand = figy / scale
            xtot = figx + x_expand

            fig_area = xtot * figy

            fig = plt.figure(figsize=(xtot, figy))
            axd = fig.subplot_mosaic(
                mosaic,
                gridspec_kw={
                    "width_ratios": [
                        x_expand / xtot,
                        figx / xtot,
                    ]
                },
            )
            ax1 = axd["A"]
            ax2 = axd["B"]
            stretch = 1 + ((scale - 2) / 1.5)
            ax1.set_xlim(-1.75, 1.75)
            ax1.set_ylim(-1.75, 1.75 * stretch)
            ax2.set_xlim(-1.75, 1.75)
            ax2.set_ylim(-1.75 * stretch, 1.75)
        else:
            mosaic = """
                CC
                AB
            """
            y_expand = figx / scale
            ytot = figy + y_expand

            fig_area = figx * ytot

            fig = plt.figure(figsize=(figx, ytot))
            axd = fig.subplot_mosaic(
                mosaic,
                gridspec_kw={
                    "height_ratios": [
                        figy / ytot,
                        y_expand / ytot,
                    ]
                },
            )
            ax1 = axd["A"]
            ax2 = axd["B"]
            stretch = 1 + ((scale - 2) / 1.5)
            ax1.set_xlim(-1.75 * stretch, 1.75)
            ax1.set_ylim(-1.75, 1.75)
            ax2.set_xlim(-1.75, 1.75 * stretch)
            ax2.set_ylim(-1.75, 1.75)

        fontsize_scale = 15 / 5.92
        linewidth_scale = 1 / 5.92
        fontsize = fontsize_scale * np.sqrt(fig_area)
        lw = linewidth_scale * np.sqrt(fig_area)

        self._setup_label_axis(ax1)
        self._setup_label_axis(ax2)

        ax = axd["C"]

        sub_a_label = self._get_miller_label(self.substrate.miller_index_a)
        sub_b_label = self._get_miller_label(self.substrate.miller_index_b)

        film_a_label = self._get_miller_label(self.film.miller_index_a)
        film_b_label = self._get_miller_label(self.film.miller_index_b)

        sub_sc_a_label = self._get_miller_label(self.substrate_a)
        sub_sc_b_label = self._get_miller_label(self.substrate_b)

        film_sc_a_label = self._get_miller_label(self.film_a)
        film_sc_b_label = self._get_miller_label(self.film_b)

        self._add_arrows(
            ax=ax1,
            matrix=sub_matrix,
            arrow_color=sub_rgb,
            color=sub_rgb + (0.05,),
            size=0.03,
            labels=[sub_a_label, sub_b_label],
            fontsize=fontsize,
            linewidth=lw * 2,
        )

        self._add_arrows(
            ax=ax2,
            matrix=film_matrix,
            arrow_color=film_rgb,
            color=film_rgb + (0.05,),
            size=0.03,
            labels=[film_a_label, film_b_label],
            fontsize=fontsize,
            linewidth=lw * 2,
        )

        for image in sub_images:
            sub_coords = (coords + image).dot(sub_matrix)
            poly = Polygon(
                xy=sub_coords[:, :2],
                closed=True,
                facecolor=sub_rgb + (substrate_alpha,),
                edgecolor=sub_rgb,
                linewidth=lw * substrate_linewidth,
                zorder=0,
            )
            ax.add_patch(poly)

        for image in film_images:
            film_coords = (coords + image).dot(film_matrix)
            poly = Polygon(
                xy=film_coords[:, :2],
                closed=True,
                facecolor=film_rgb + (film_alpha,),
                edgecolor=film_rgb,
                linewidth=lw * film_linewidth,
                zorder=10,
            )
            ax.add_patch(poly)

        grid_x = np.linspace(0, 1, 11)
        grid_y = np.linspace(0, 1, 11)

        X, Y = np.meshgrid(grid_x, grid_y)

        sc_shifts = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0],
            ]
        )

        for shift in sc_shifts:
            shift_coords = (coords + shift).dot(interface_matrix)
            poly = Polygon(
                xy=shift_coords[:, :2],
                closed=True,
                facecolor="white",
                edgecolor="white",
                linewidth=lw * 1,
                zorder=15,
            )
            ax.add_patch(poly)

        ax.plot(
            interface_coords[:, 0],
            interface_coords[:, 1],
            color="black",
            linewidth=lw * 3,
            zorder=20,
        )

        x_range = interface_coords[:, 0].max() - interface_coords[:, 0].min()
        y_range = interface_coords[:, 1].max() - interface_coords[:, 1].min()
        x_margin = 0.15 * x_range
        y_margin = 0.15 * y_range

        ax.set_xlim(
            interface_coords[:, 0].min() - x_margin,
            interface_coords[:, 0].max() + x_margin,
        )
        ax.set_ylim(
            interface_coords[:, 1].min() - y_margin,
            interface_coords[:, 1].max() + y_margin,
        )

        xlabel = " ".join(
            [
                f"{int(self._film_supercell_scale_factors[0])}",
                film_sc_a_label,
                "_{F}",
                "||",
                f"{int(self._substrate_supercell_scale_factors[0])}",
                sub_sc_a_label,
                "_{S}",
            ]
        )

        ylabel = " ".join(
            [
                f"{int(self._film_supercell_scale_factors[1])}",
                film_sc_b_label,
                "_{F}",
                "||",
                f"{int(self._substrate_supercell_scale_factors[1])}",
                sub_sc_b_label,
                "_{S}",
            ]
        )

        t_not_show = ax.text(
            0.5 * interface_matrix[0][0],
            0.5 * interface_matrix[0][1],
            "$" + xlabel + "$",
            fontsize=fontsize,
            ha="center",
            va="center",
            zorder=100,
        )

        fig.canvas.draw()
        bbox = ax.transData.inverted().transform_bbox(
            t_not_show.get_window_extent(fig.canvas.get_renderer())
        )
        height = 0.75 * bbox.height
        t_not_show.remove()

        x_label_shift = self._add_sc_labels(
            ax,
            label=xlabel,
            vector=interface_matrix[0],
            fontsize=fontsize,
            rotation=0,
            linewidth=lw * 2,
            height=height,
        )
        y_label_shift = self._add_sc_labels(
            ax,
            label=ylabel,
            vector=interface_matrix[1],
            fontsize=fontsize,
            rotation=np.rad2deg(self.match.substrate_angle),
            linewidth=lw * 2,
            height=height,
        )

        x_range = interface_coords[:, 0].max() - interface_coords[:, 0].min()
        y_range = interface_coords[:, 1].max() - interface_coords[:, 1].min()
        x_margin = 0.05 * x_range
        y_margin = 0.05 * y_range

        ax.set_xlim(
            interface_coords[:, 0].min() + (2 * y_label_shift[0]) - x_margin,
            interface_coords[:, 0].max() + x_margin,
        )
        ax.set_ylim(
            interface_coords[:, 1].min() + (2 * x_label_shift[1]) - y_margin,
            interface_coords[:, 1].max() + (2 * y_label_shift[1]) + y_margin,
        )

        ax.tick_params(
            left=False, labelleft=False, bottom=False, labelbottom=False
        )
        ax.spines[:].set_visible(False)

        ax.set_aspect("equal")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")

        if not show_in_colab:
            plt.close()

    # def plot_interface_old(
    #     self,
    #     output: str = "interface_view.png",
    #     strain: bool = True,
    #     dpi: int = 400,
    #     show_in_colab: bool = False,
    # ) -> None:
    #     """
    #     This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

    #     Args:
    #         output: File path for the output image
    #         strain: Determines if the film lattice should be strained so it shows perfectly aligned lattice coincidence sites,
    #             or if the film lattice should be unstrained, giving a better visual of the lattice mismatch.
    #         dpi: dpi (dots per inch) of the output image.
    #             Setting dpi=100 gives reasonably sized images when viewed in colab notebook
    #         show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
    #             if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
    #     """
    #     sub_struc = self.substrate._orthogonal_slab_structure.copy()
    #     sub_a_to_i_op = SymmOp.from_rotation_and_translation(
    #         rotation_matrix=self._substrate_a_to_i, translation_vec=np.zeros(3)
    #     )
    #     sub_struc.apply_operation(sub_a_to_i_op)

    #     film_struc = self.film._orthogonal_slab_structure.copy()
    #     film_a_to_i_op = SymmOp.from_rotation_and_translation(
    #         rotation_matrix=self._film_a_to_i, translation_vec=np.zeros(3)
    #     )
    #     film_struc.apply_operation(film_a_to_i_op)

    #     sub_matrix = sub_struc.lattice.matrix
    #     film_matrix = film_struc.lattice.matrix

    #     sub_sc_matrix = deepcopy(self._substrate_supercell.lattice.matrix)
    #     film_sc_matrix = deepcopy(self._film_supercell.lattice.matrix)

    #     coords = np.array(
    #         [
    #             [0, 0, 0],
    #             [1, 0, 0],
    #             [1, 1, 0],
    #             [0, 1, 0],
    #             [0, 0, 0],
    #         ]
    #     )

    #     sc_shifts = np.array(
    #         [
    #             [0, 0, 0],
    #             [1, 0, 0],
    #             [0, 1, 0],
    #             [-1, 0, 0],
    #             [0, -1, 0],
    #             [1, 1, 0],
    #             [-1, -1, 0],
    #             [1, -1, 0],
    #             [-1, 1, 0],
    #         ]
    #     )

    #     sub_sc_shifts = sc_shifts.dot(sub_sc_matrix)
    #     film_sc_shifts = sc_shifts.dot(film_sc_matrix)
    #     sub_sl = coords.dot(sub_sc_matrix)

    #     sub_struc, sub_inv_matrix = self._generate_sc_for_interface_view(
    #         struc=sub_struc,
    #         transformation_matrix=self.match.substrate_sl_transform,
    #     )

    #     film_struc, film_inv_matrix = self._generate_sc_for_interface_view(
    #         struc=film_struc,
    #         transformation_matrix=self.match.film_sl_transform,
    #     )

    #     fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    #     for c in sub_struc.cart_coords:
    #         for shift in sub_sc_shifts:
    #             self._plot_interface_view(
    #                 ax=ax,
    #                 zero_coord=c,
    #                 supercell_shift=shift,
    #                 cell_vetices=coords,
    #                 slab_matrix=sub_matrix,
    #                 sc_inv_matrix=sub_inv_matrix,
    #                 is_film=False,
    #                 strain=strain,
    #                 facecolor=(0, 0, 1, 0.2),
    #                 edgecolor=(0, 0, 1, 1),
    #             )

    #     for c in film_struc.cart_coords:
    #         for shift in film_sc_shifts:
    #             self._plot_interface_view(
    #                 ax=ax,
    #                 zero_coord=c,
    #                 supercell_shift=shift,
    #                 cell_vetices=coords,
    #                 slab_matrix=film_matrix,
    #                 sc_inv_matrix=film_inv_matrix,
    #                 is_film=True,
    #                 strain=strain,
    #                 facecolor=(200 / 255, 0, 0, 0.2),
    #                 edgecolor=(200 / 255, 0, 0, 1),
    #             )

    #     ax.plot(
    #         sub_sl[:, 0],
    #         sub_sl[:, 1],
    #         color="black",
    #         linewidth=3,
    #     )

    #     ax.set_aspect("equal")
    #     ax.axis("off")

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")

    #     if not show_in_colab:
    #         plt.close()
