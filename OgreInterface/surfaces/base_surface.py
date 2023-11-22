"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from abc import ABC, abstractmethod
from typing import Dict, Union, Iterable, List, Tuple, TypeVar
import itertools
from copy import deepcopy
import warnings

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import numpy as np
from ase import Atoms

from OgreInterface import utils
from OgreInterface.surfaces.oriented_bulk import OrientedBulk


# suppress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")

SelfBaseSurface = TypeVar("SelfBaseSurface", bound="BaseSurface")


class BaseSurface(ABC):
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
        slab: Structure,
        oriented_bulk: OrientedBulk,
        miller_index: list,
        layers: int,
        vacuum: float,
        termination_index: int,
    ) -> None:
        self.oriented_bulk = oriented_bulk
        self._non_orthogonal_slab_structure = slab
        self._orthogonal_slab_structure = self._orthogonalize_slab(slab)
        self.oriented_bulk_structure = (
            self.oriented_bulk.oriented_bulk_structure
        )
        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.termination_index = termination_index
        self._passivated = False

    def _orthogonalize_slab(self, non_orthogonal_slab: Structure) -> Structure:
        """
        This function is used to force a slab's c-vector to be orthogonal to
        the a- and b-lattice vector

        Args:
            non_orthogonal_slab: Slab structure with a non-orthogonal c-vector

        Returns:
            An orthogonalized slab structure
        """
        # Get the a, b, and c lattice vectors
        a, b, c = non_orthogonal_slab.lattice.matrix

        # The orthogonal c-vector is the unit normal vector of the surface
        # multiplied by the projected length of the non-orthogonal c-vector
        # on the unit surface normal
        new_c = (
            np.dot(c, self.oriented_bulk.surface_normal)
            * self.oriented_bulk.surface_normal
        )

        # Create the new lattice matrix and the orthogonal structure
        orthogonal_matrix = np.vstack([a, b, new_c])
        orthogonal_slab = Structure(
            lattice=Lattice(matrix=orthogonal_matrix),
            species=non_orthogonal_slab.species,
            coords=non_orthogonal_slab.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=non_orthogonal_slab.site_properties,
        )
        orthogonal_slab.sort()

        # Shift the structure so the top atom is inline with the c-vector
        # This is mostly for aesthetic purposes so the orthogonal and
        # non-orthogonal slabs appear the same.
        top_z = non_orthogonal_slab.frac_coords[:, -1].max()
        top_cart = non_orthogonal_slab.lattice.matrix[-1] * top_z
        top_frac = orthogonal_slab.lattice.get_fractional_coords(top_cart)
        top_frac[-1] = 0.0
        orthogonal_slab.translate_sites(
            indices=range(len(orthogonal_slab)),
            vector=-top_frac,
            frac_coords=True,
            to_unit_cell=True,
        )

        # Round and mod the structure
        orthogonal_slab = utils.get_rounded_structure(
            structure=orthogonal_slab,
            tol=6,
        )

        return orthogonal_slab

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
            return utils.return_structure(
                structure=self._orthogonal_slab_structure,
                convert_to_atoms=return_atoms,
            )
        else:
            return utils.return_structure(
                structure=self._non_orthogonal_slab_structure,
                convert_to_atoms=return_atoms,
            )

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
    def surface_normal(self) -> np.ndarray:
        return self.oriented_bulk.surface_normal

    @property
    def layer_thickness(self) -> float:
        return self.oriented_bulk.layer_thickness

    @property
    def bulk_structure(self) -> Structure:
        return self.oriented_bulk.bulk

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
            self.vacuum / self.oriented_bulk.layer_thickness
        )

        return (layer_mat @ self.oriented_bulk.transformation_matrix).astype(
            int
        )

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
        return self.oriented_bulk.transformation_matrix.astype(int)

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
        return self.oriented_bulk._init_bulk.composition.reduced_formula

    @property
    def formula_with_miller(self) -> str:
        """
        Reduced formula of the surface and the miller index added

        Examples:
            >>> surface.latex_formula
            >>> "CsPbBr3(1-10)

        Returns:
            Reduced formula of the underlying bulk structure with the miller index
        """
        return (
            f"{self.formula}({''.join([str(i) for i in self.miller_index])})"
        )

    @property
    def latex_formula(self) -> str:
        """
        Reduced formula of the surface formatted with latex

        Examples:
            >>> surface.latex_formula
            >>> "CsPbBr$_{3}$"

        Returns:
            Reduced formula of the underlying bulk structure where subscripts are formated for latex
        """
        return utils.get_latex_formula(self.formula)

    @property
    def latex_formula_with_miller(self) -> str:
        """
        Reduced formula of the surface formatted with latex and the miller index added

        Examples:
            >>> surface.latex_formula
            >>> "CsPbBr$_{3}$"(1$\\overline{1}$0)

        Returns:
            Reduced formula of the underlying bulk structure with the miller index where subscripts are formated for latex
        """
        return f"{self.latex_formula}({utils.get_miller_index_label(self.miller_index)})"

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
        return self.oriented_bulk.crystallographic_basis[0].astype(int)

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
        return self.oriented_bulk.crystallographic_basis[1].astype(int)

    @property
    def crystallographic_basis(self) -> np.ndarray:
        return self.oriented_bulk.crystallographic_basis

    @abstractmethod
    def write_file(self, *args, **kwargs) -> None:
        pass

    def _get_base_poscar_comment_str(self, orthogonal: bool):
        comment = "|".join(
            [
                f"L={self.layers}",
                f"T={self.termination_index}",
                f"O={orthogonal}",
            ]
        )

        return comment

    def remove_layers(
        self,
        num_layers: int,
        atomic_layers: bool = True,
        top: bool = False,
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
        if top:
            if atomic_layers:
                total_layers = self.atomic_layers
            else:
                total_layers = self.layers

            layer_inds = total_layers - np.arange(num_layers) - 1
        else:
            layer_inds = np.arange(num_layers)

        to_delete = []
        for layer in layer_inds:
            atom_inds = self.get_layer_indices(
                layer=layer,
                atomic_layers=atomic_layers,
            )
            to_delete.append(atom_inds)

        to_delete = np.concatenate(to_delete)

        self._orthogonal_slab_structure.remove_sites(to_delete)
        self._non_orthogonal_slab_structure.remove_sites(to_delete)

    def get_termination(self):
        """
        Returns the termination of the surface as a dictionary

        Examples:
            >>> surface.get_termination()
            >>> {"bottom": {"In": 1, "As": 0}, "top": {"In": 0, "As": 1}
        """
        raise NotImplementedError
