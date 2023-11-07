import typing as tp

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
from ase import Atoms
import numpy as np

from OgreInterface.interfaces.base_interface import BaseInterface
from OgreInterface.lattice_match import OgreMatch
from OgreInterface.surfaces import Surface
from OgreInterface import utils

SelfMolecularInterface = tp.TypeVar(
    "SelfMolecularInterface", bound="MolecularInterface"
)


class MolecularInterface(BaseInterface):
    def __init__(
        self,
        substrate: tp.Union[Surface, SelfMolecularInterface],
        film: tp.Union[Surface, SelfMolecularInterface],
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
        substrate_strain_fraction: float = 0.0,
    ) -> SelfMolecularInterface:
        super().__init__(
            substrate=substrate,
            film=film,
            match=match,
            interfacial_distance=interfacial_distance,
            vacuum=vacuum,
            center=center,
            substrate_strain_fraction=substrate_strain_fraction,
        )

    def write_file(
        self,
        output: str = "POSCAR_interface",
        orthogonal: bool = True,
    ) -> None:
        """
        Write the POSCAR of the interface

        Args:
            output: File path of the output POSCAR
            orthogonal: Determines of the orthogonal structure is written
        """
        if orthogonal:
            slab = utils.return_structure(
                structure=self._orthogonal_structure,
                convert_to_atoms=False,
            )
        else:
            slab = utils.return_structure(
                structure=self._non_orthogonal_structure,
                convert_to_atoms=False,
            )

        comment = self._get_base_poscar_comment_str(orthogonal=orthogonal)

        comment += "|" + "|".join(
            [
                f"a={self._a_shift:.4f}",
                f"b={self._b_shift:.4f}",
            ]
        )

        poscar = Poscar(slab, comment=comment)
        poscar.write_file(output)
