"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from typing import TypeVar, Union, List, Optional

from pymatgen.core.structure import Structure
from ase import Atoms

from OgreInterface.generate.base_surface_generator import BaseSurfaceGenerator
from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.surfaces.surface import Surface

SelfSurfaceGenerator = TypeVar(
    "SelfSurfaceGenerator", bound="SurfaceGenerator"
)


class SurfaceGenerator(BaseSurfaceGenerator):
    """Class for generating surfaces from a given bulk structure.

    The SurfaceGenerator classes generates surfaces with all possible terminations and contains
    information pertinent to generating interfaces with the InterfaceGenerator.

    Examples:
        Creating a SurfaceGenerator object using PyMatGen to load the structure:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> from pymatgen.core.structure import Structure
        >>> bulk = Structure.from_file("POSCAR_bulk")
        >>> surfaces = SurfaceGenerator(bulk=bulk, miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

        Creating a SurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

    Args:
        bulk: Bulk crystal structure used to create the surface
        miller_index: Miller index of the surface
        layers: Number of layers to include in the surface
        minimum_thickness: Optional flag to set the minimum thickness of the slab. If this is not None, then it will override the layers value
        vacuum: Size of the vacuum to include over the surface in Angstroms
        refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
            This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
            users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
            work exclusively with the primitive sturcture so we always have it on hand.
        generate_all: Determines if all possible surface terminations are generated.
        lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)
        suppress_warnings: This gives the user the option to suppress warnings if they know what they are doing and don't need to see the warning messages

    Attributes:
        slabs (list): List of OgreInterface Surface objects with different surface terminations.
        bulk_structure (Structure): Pymatgen Structure class for the conventional cell of the input bulk structure
        bulk_atoms (Atoms): ASE Atoms class for the conventional cell of the input bulk structure
        primitive_structure (Structure): Pymatgen Structure class for the primitive cell of the input bulk structure
        primitive_atoms (Atoms): ASE Atoms class for the primitive cell of the input bulk structure
        miller_index (list): Miller index of the surface
        layers (int): Number of layers to include in the surface
        vacuum (float): Size of the vacuum to include over the surface in Angstroms
        generate_all (bool): Determines if all possible surface terminations are generated.
        lazy (bool): Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)
        oriented_bulk_structure (Structure): Pymatgen Structure class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        oriented_bulk_atoms (Atoms): Pymatgen Atoms class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        uvw_basis (list): The miller indices of the slab lattice vectors.
        transformation_matrix: Transformation matrix used to convert from the bulk basis to the slab basis
            (usefull for band unfolding calculations)
        inplane_vectors (list): The cartesian vectors of the in-plane lattice vectors.
        surface_normal (list): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        bulk: Union[Structure, Atoms],
        miller_index: List[int],
        layers: Optional[int] = None,
        minimum_thickness: Optional[float] = 18.0,
        vacuum: float = 40.0,
        refine_structure: bool = True,
        make_planar: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
        layer_grouping_tolarence: Optional[float] = None,
    ) -> SelfSurfaceGenerator:
        super().__init__(
            bulk=bulk,
            miller_index=miller_index,
            surface_type=Surface,
            layers=layers,
            minimum_thickness=minimum_thickness,
            vacuum=vacuum,
            refine_structure=refine_structure,
            make_planar=make_planar,
            generate_all=generate_all,
            lazy=lazy,
            suppress_warnings=suppress_warnings,
            layer_grouping_tolarence=layer_grouping_tolarence,
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        miller_index: List[int],
        layers: Optional[int] = None,
        minimum_thickness: Optional[float] = 18.0,
        vacuum: float = 40.0,
        refine_structure: bool = True,
        make_planar: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
        layer_grouping_tolarence: Optional[float] = None,
    ) -> SelfSurfaceGenerator:
        return super().from_file(
            filename=filename,
            miller_index=miller_index,
            layers=layers,
            minimum_thickness=minimum_thickness,
            vacuum=vacuum,
            refine_structure=refine_structure,
            make_planar=make_planar,
            generate_all=generate_all,
            lazy=lazy,
            suppress_warnings=suppress_warnings,
            layer_grouping_tolarence=layer_grouping_tolarence,
        )

    def _get_slab_base(self) -> OrientedBulk:
        self.obs.add_charges()
        return self.obs
