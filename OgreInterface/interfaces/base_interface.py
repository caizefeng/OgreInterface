"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from __future__ import annotations
import typing as tp
from itertools import combinations, groupby
from copy import deepcopy
from functools import reduce
import warnings
from abc import abstractproperty, abstractmethod, ABC

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import CrystalNN, BrunnerNN_real
import numpy as np
from ase import Atoms

from OgreInterface import utils
from OgreInterface.lattice_match import OgreMatch
from OgreInterface.plotting_tools import plot_match
from OgreInterface.surfaces.surface import Surface
from OgreInterface.surfaces.base_surface import BaseSurface
from OgreInterface.surfaces.molecular_surface import MolecularSurface

if tp.TYPE_CHECKING:
    from OgreInterface.interfaces.interface import Interface
    from OgreInterface.interfaces.molecular_interface import MolecularInterface


# suppress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")

SelfBaseInterface = tp.TypeVar("SelfBaseInterface", bound="BaseInterface")


class BaseInterface(ABC):
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
        substrate: tp.Union[
            Surface,
            MolecularSurface,
            Interface,
            MolecularInterface,
        ],
        film: tp.Union[
            Surface,
            MolecularSurface,
            Interface,
            MolecularInterface,
        ],
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
        substrate_strain_fraction: float = 0.0,
    ) -> None:
        self.center = center
        self.substrate = substrate
        self.film = film
        self.match = match
        self.vacuum = vacuum
        self._substrate_strain_fraction = substrate_strain_fraction

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

        self._substrate_a_to_i = self.match.substrate_align_transform.T
        self._film_a_to_i = self.match.film_align_transform.T

        self._orient_structure(
            structure=self._substrate_supercell,
            transform=self._substrate_a_to_i,
        )

        self._orient_structure(
            structure=self._film_supercell,
            transform=self._film_a_to_i,
        )

        self._avg_sc_inplane_lattice = self._get_average_inplane_lattice()

        self.interfacial_distance = interfacial_distance
        self._a_shift = 0.0
        self._b_shift = 0.0

        (
            self._strained_sub,
            self._substrate_strain_matrix,
        ) = self._prepare_substrate()

        (
            self._strained_film,
            self._film_strain_matrix,
        ) = self._prepare_film()

        (
            self._M_matrix,
            self._non_orthogonal_structure,
            self._non_orthogonal_substrate_structure,
            self._non_orthogonal_film_structure,
            self._orthogonal_structure,
            self._orthogonal_substrate_structure,
            self._orthogonal_film_structure,
        ) = self._stack_interface()

    def _get_average_inplane_lattice(self):
        film_lattice = self._film_supercell.lattice.matrix[:2]
        substrate_lattice = self._substrate_supercell.lattice.matrix[:2]

        film_frac = self._substrate_strain_fraction
        sub_frac = 1 - film_frac

        avg_lattice = (film_frac * film_lattice) + (
            sub_frac * substrate_lattice
        )

        return avg_lattice

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
        return utils.return_structure(
            structure=self.substrate.oriented_bulk_structure,
            convert_to_atoms=False,
        )

    @property
    def substrate_oriented_bulk_supercell(self) -> Structure:
        if self._substrate_obs_supercell is not None:
            obs_supercell = utils.apply_strain_matrix(
                structure=self._substrate_obs_supercell,
                strain_matrix=self._substrate_strain_matrix,
            )

            return utils.return_structure(
                structure=obs_supercell,
                convert_to_atoms=False,
            )
        else:
            raise "substrate_oriented_bulk_supercell is not applicable when an Interface is used as the substrate"

    @property
    def film_oriented_bulk_supercell(self) -> Structure:
        if self._film_obs_supercell is not None:
            obs_supercell = utils.apply_strain_matrix(
                structure=self._film_obs_supercell,
                strain_matrix=self._film_strain_matrix,
            )

            return utils.return_structure(
                structure=obs_supercell,
                convert_to_atoms=False,
            )
        else:
            raise "film_oriented_bulk_supercell is not applicable when an Interface is used as the film"

    @property
    def substrate_oriented_bulk_structure(self) -> Structure:
        if issubclass(type(self.substrate), BaseSurface):
            obs_structure = utils.apply_strain_matrix(
                structure=self.substrate.oriented_bulk_structure,
                strain_matrix=self._substrate_strain_matrix,
            )

            self._orient_structure(
                structure=obs_structure,
                transform=self._substrate_a_to_i,
            )

            return utils.return_structure(
                structure=obs_structure,
                convert_to_atoms=False,
            )
        else:
            raise "substrate_oriented_bulk_structure is not applicable when an Interface is used as the substrate"

    @property
    def film_oriented_bulk_structure(self) -> Structure:
        if issubclass(type(self.film), BaseSurface):
            obs_structure = utils.apply_strain_matrix(
                structure=self.film.oriented_bulk_structure,
                strain_matrix=self._film_strain_matrix,
            )

            self._orient_structure(
                structure=obs_structure,
                transform=self._film_a_to_i,
            )

            return utils.return_structure(
                structure=obs_structure,
                convert_to_atoms=False,
            )
        else:
            raise "film_oriented_bulk_structure is not applicable when an Interface is used as the film"

    @property
    def layer_thickness(self) -> float:
        return self.substrate.oriented_bulk.layer_thickness

    def _passivated(self) -> bool:
        return self.substrate._passivated or self.film._passivated

    @property
    def bulk_transformation_matrix(self) -> np.ndarray:
        return self.substrate.bulk_transformation_matrix

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

    @property
    def interface_height(self):
        """
        This returns the z-height of the interface (average between the top film atom z and bottom substrate z)
        """
        sub_z_coords = self._orthogonal_substrate_structure.cart_coords[:, -1]
        film_z_coords = self._orthogonal_film_structure.cart_coords[:, -1]

        return (sub_z_coords.min() + film_z_coords.max()) / 2

    def get_interface(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the interface structure generated from the OgreMatch

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the interface structure
        """
        if orthogonal:
            return utils.return_structure(
                structure=self._orthogonal_structure,
                convert_to_atoms=return_atoms,
            )
        else:
            return utils.return_structure(
                structure=self._non_orthogonal_structure,
                convert_to_atoms=return_atoms,
            )

    def get_substrate_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
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
            return utils.return_structure(
                structure=self._orthogonal_substrate_structure,
                convert_to_atoms=return_atoms,
            )
        else:
            return utils.return_structure(
                structure=self._non_orthogonal_substrate_structure,
                convert_to_atoms=return_atoms,
            )

    def get_film_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
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
            return utils.return_structure(
                structure=self._orthogonal_film_structure,
                convert_to_atoms=return_atoms,
            )
        else:
            return utils.return_structure(
                structure=self._non_orthogonal_film_structure,
                convert_to_atoms=return_atoms,
            )

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

        if "molecules" in interface.site_properties:
            interface = utils.add_molecules(interface)

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

        if "molecules" in interface.site_properties:
            interface = utils.add_molecules(interface)

        site_props = interface.site_properties
        is_film = np.array(site_props["is_film"])
        layer_index = np.array(site_props[layer_key])
        is_layer = layer_index == layer_from_interface

        return np.where(np.logical_and(is_film, is_layer))[0]

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
            "Strain (%): " + f"{100*self.match.strain:.3f}",
            "Cross Section Area (Ang^2): " + f"{self.area:.3f}",
        ]
        return_str = "\n".join(return_info)

        return return_str

    def _shift_film(
        self,
        interface: Structure,
        shift: tp.Iterable,
        fractional: bool,
    ) -> tp.Tuple[Structure, Structure]:
        shifted_interface_structure = utils.shift_film(
            interface=interface,
            shift=shift,
            fractional=fractional,
        )

        (
            shifted_film_structure,
            _,
        ) = self._get_film_and_substrate_parts(shifted_interface_structure)

        return (
            shifted_interface_structure,
            shifted_film_structure,
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
            self._orthogonal_film_structure,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=shift,
            fractional=False,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_film_structure,
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
            self._orthogonal_film_structure,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_film_structure,
        ) = self._shift_film(
            interface=self._non_orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )

    def _create_supercell(
        self, substrate: bool = True
    ) -> tp.Tuple[Structure, Structure, np.ndarray, np.ndarray]:
        if substrate:
            matrix = self.match.substrate_sl_transform

            if issubclass(type(self.substrate), BaseSurface):
                supercell = (
                    self.substrate._non_orthogonal_slab_structure.copy()
                )
                obs_supercell = self.substrate.oriented_bulk_structure.copy()
            elif type(self.substrate) is Interface:
                supercell = self.substrate._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    not_hydrogen = layer_index != -1
                    is_film = np.array(supercell.site_properties["is_film"])
                    is_sub = np.array(supercell.site_properties["is_sub"])
                    layer_index[(is_film & not_hydrogen)] += (
                        layer_index[is_sub].max() + 1
                    )
                    supercell.add_site_property(
                        layer_key,
                        layer_index.tolist(),
                    )

            basis = self.substrate.crystallographic_basis
        else:
            matrix = self.match.film_sl_transform

            if issubclass(type(self.film), BaseSurface):
                supercell = self.film._non_orthogonal_slab_structure.copy()
                obs_supercell = self.film.oriented_bulk_structure.copy()
            elif type(self.film) is Interface:
                supercell = self.film._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    is_film = np.array(supercell.site_properties["is_film"])
                    is_sub = np.array(supercell.site_properties["is_sub"])
                    layer_index[is_film] += layer_index[is_sub].max() + 1
                    supercell.add_site_property(
                        layer_key,
                        layer_index.tolist(),
                    )

            basis = self.film.crystallographic_basis

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

    def _orient_structure(
        self,
        structure: Structure,
        transform: np.ndaarray,
    ) -> None:
        op = SymmOp.from_rotation_and_translation(
            transform,
            translation_vec=np.zeros(3),
        )

        if "molecules" in structure.site_properties:
            utils.apply_op_to_mols(structure, op)

        structure.apply_operation(op)

    def _prepare_film(self) -> Structure:
        supercell_slab = self._film_supercell
        sc_matrix = supercell_slab.lattice.matrix
        sub_sc_matrix = self._strained_sub.lattice.matrix

        inplane_strain_transformation = (
            np.linalg.inv(sc_matrix[:2, :2]) @ sub_sc_matrix[:2, :2]
        )
        inplane_strained_matrix = (
            sc_matrix[:, :2] @ inplane_strain_transformation
        )

        strained_matrix = np.c_[inplane_strained_matrix, sc_matrix[:, -1]]

        init_volume = supercell_slab.volume
        strain_volume = np.abs(np.linalg.det(strained_matrix))
        scale_factor = init_volume / strain_volume

        # Maintain constant volume
        strained_matrix[-1, -1] *= scale_factor
        const_volume_strain_transformation = (
            np.linalg.inv(sc_matrix) @ strained_matrix
        )
        strained_film = utils.apply_strain_matrix(
            structure=supercell_slab,
            strain_matrix=const_volume_strain_transformation,
        )

        sub_non_orth_c_vec = self._strained_sub.lattice.matrix[-1]
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

        return oriented_film, const_volume_strain_transformation

    def _prepare_substrate(self) -> Structure:
        supercell_slab = self._substrate_supercell
        sc_matrix = supercell_slab.lattice.matrix
        avg_sc_matrix = self._avg_sc_inplane_lattice

        inplane_strain_transformation = (
            np.linalg.inv(sc_matrix[:2, :2]) @ avg_sc_matrix[:2, :2]
        )

        inplane_strained_matrix = (
            sc_matrix[:, :2] @ inplane_strain_transformation
        )

        strained_matrix = np.c_[inplane_strained_matrix, sc_matrix[:, -1]]

        init_volume = supercell_slab.volume
        strain_volume = np.abs(np.linalg.det(strained_matrix))
        scale_factor = init_volume / strain_volume

        # Maintain constant volume
        strained_matrix[-1, -1] *= scale_factor
        const_volume_strain_transformation = (
            np.linalg.inv(sc_matrix) @ strained_matrix
        )
        strained_substrate = utils.apply_strain_matrix(
            structure=supercell_slab,
            strain_matrix=const_volume_strain_transformation,
        )

        return strained_substrate, const_volume_strain_transformation

    def _stack_interface(
        self,
    ) -> tp.Tuple[
        np.ndarray,
        Structure,
        Structure,
        Structure,
        Structure,
        Structure,
        Structure,
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
        c_norm_proj = self.substrate.layer_thickness / oriented_bulk_c

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
        sub_M = self.substrate.bulk_transformation_matrix
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
        interface_site_properties["is_sub"] = (
            np.array([True] * len(strained_sub) + [False] * len(strained_film))
            .astype(bool)
            .tolist()
        )
        interface_site_properties["is_film"] = (
            np.array([False] * len(strained_sub) + [True] * len(strained_film))
            .astype(bool)
            .tolist()
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
            ortho_sub_structure,
        ) = self._get_film_and_substrate_parts(ortho_interface_struc)
        (
            non_ortho_film_structure,
            non_ortho_sub_structure,
        ) = self._get_film_and_substrate_parts(non_ortho_interface_struc)

        return (
            interface_M,
            non_ortho_interface_struc,
            non_ortho_sub_structure,
            non_ortho_film_structure,
            ortho_interface_struc,
            ortho_sub_structure,
            ortho_film_structure,
        )

    def _get_film_and_substrate_parts(
        self,
        interface: Structure,
    ) -> tp.Tuple[Structure, Structure]:
        film_inds = np.where(interface.site_properties["is_film"])[0]
        sub_inds = np.where(interface.site_properties["is_sub"])[0]

        film_structure = interface.copy()
        film_structure.remove_sites(sub_inds)

        sub_structure = interface.copy()
        sub_structure.remove_sites(film_inds)

        return film_structure, sub_structure

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

    def plot_interface(
        self,
        output: str = "interface_view.png",
        dpi: int = 400,
        film_color: str = "green",
        substrate_color: str = "orange",
        show_in_colab: bool = False,
    ) -> None:
        """
        This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

        Args:
            output: File path for the output image
            dpi: dpi (dots per inch) of the output image.
                Setting dpi=100 gives reasonably sized images when viewed in colab notebook
            film_color: Color to represent the film lattice vectors
            substrate_color: Color to represent the substrate lattice vectors
            show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
                if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
        """
        plot_match(
            match=self.match,
            padding=0.2,
            substrate_color=substrate_color,
            film_color=film_color,
            output=output,
            show_in_colab=show_in_colab,
        )

    def _get_base_poscar_comment_str(self, orthogonal: bool):
        comment = "|".join(
            [
                f"L={self.film.layers},{self.substrate.layers}",
                f"T={self.film.termination_index},{self.substrate.termination_index}",
                f"O={int(orthogonal)}",
                f"d={self.interfacial_distance:.3f}",
            ]
        )

        return comment
