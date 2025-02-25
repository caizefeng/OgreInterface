import typing as tp

import numpy as np
from pymatgen.core.structure import Structure

from OgreInterface import utils
from OgreInterface.surface_matching import BaseSurfaceEnergy
from OgreInterface.surface_matching.mlip_surface_matcher.mlip_calculator_factory import get_combined_calculator
from OgreInterface.surface_matching.mlip_surface_matcher.utils import structure_to_atoms
from OgreInterface.surfaces import Surface


class MLIPSurfaceEnergy(BaseSurfaceEnergy):
    def __init__(
            self,
            surface: Surface,
            mlip: str = "MACE",
            model_name: tp.Optional[str] = None,
            device: str = "cuda",
    ):
        """
        Initialize the MLIPSurfaceEnergy with a chosen MLIP and DFT-D3.

        Args:
            surface (Surface): Surface structure.
            mlip (str): Choice of MLIP model. Options: "CHGNet", "MACE", "DP".
            model_name (Optional[str]): Model name (if using MACE or DP).
            device (str): Compute device ("cuda" or "cpu").
        """
        super().__init__(surface=surface)

        # Get the combined calculator using the factory helper
        self.calculator = get_combined_calculator(mlip=mlip,
                                                  model_name=model_name,
                                                  device=device)

    def generate_constant_inputs(self, structure: Structure) -> tp.List[Structure]:
        """Generate constant structure inputs (e.g., bulk, supercells)."""
        return [structure]

    def generate_interface_inputs(self, shifts: np.ndarray) -> tp.List[Structure]:
        """Generate interface structures given shifts."""
        shifted_ifaces = [
            utils.shift_film(interface=self.double_slab, shift=shift, fractional=False)
            for shift in shifts
        ]
        return shifted_ifaces

    def calculate(self, inputs: tp.List[Structure]) -> np.ndarray:
        """
        Compute total energy using the selected MLIP model and DFT-D3.

        Args:
            inputs (List[Structure]): List of pymatgen structures.

        Returns:
            np.ndarray: Array of total energies.
        """
        energies = []
        for structure in inputs:
            atoms = structure_to_atoms(structure)  # Convert structure manually
            atoms.set_calculator(self.calculator)
            energy = atoms.get_potential_energy()
            energies.append(energy)

        return np.array(energies)
