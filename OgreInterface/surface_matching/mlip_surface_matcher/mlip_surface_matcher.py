import typing as tp

import numpy as np
from pymatgen.core.structure import Structure

from OgreInterface import utils
from OgreInterface.interfaces import Interface
from OgreInterface.surface_matching import (
    BaseSurfaceMatcher,
    MLIPSurfaceEnergy,
)
from OgreInterface.surface_matching.mlip_surface_matcher.mlip_calculator_factory import get_combined_calculator
from OgreInterface.surface_matching.mlip_surface_matcher.utils import structure_to_atoms


class MLIPSurfaceMatcher(BaseSurfaceMatcher):
    def __init__(
            self,
            interface: Interface,
            mlip: str = "MACE",
            model_name: tp.Optional[str] = None,
            grid_density: float = 2.5,
            device: str = "cuda",
            verbose: bool = True,
    ):
        """
        Initialize the MLIPSurfaceMatcher with a chosen machine-learning interatomic potential.

        Args:
            interface (Interface): Interface structure.
            mlip (str): Choice of MLIP model. Options: "CHGNet", "MACE", "DP".
            model_name (Optional[str]): Model name (if using MACE or DP).
            grid_density (float): Grid density for surface matching.
            device (str): Compute device ("cuda" or "cpu").
        """
        super().__init__(interface=interface, grid_density=grid_density, verbose=verbose)

        # Get the combined calculator using the factory helper
        self.calculator = get_combined_calculator(mlip=mlip,
                                                  model_name=model_name,
                                                  device=device)

        self.surface_energy_kwargs = {"mlip": mlip,
                                      "model_name": model_name,
                                      "device": device, }

    def generate_constant_inputs(self, structure: Structure) -> tp.List[Structure]:
        """Generate constant structure inputs (e.g., bulk, supercells)."""
        return [structure]

    def generate_interface_inputs(self, shifts: np.ndarray) -> tp.List[Structure]:
        """Generate interface structures given shifts."""
        shifted_ifaces = [
            utils.shift_film(interface=self.iface, shift=shift, fractional=False)
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

    @property
    def surface_energy_module(self) -> MLIPSurfaceEnergy:
        """
        Set the surface energy module here
        """
        return MLIPSurfaceEnergy
