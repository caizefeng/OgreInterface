import typing as tp

from pymatgen.core.structure import Structure
import numpy as np

from OgreInterface.surface_matching import (
    BaseSurfaceMatcher,
    CHGNetSurfaceEnergy,
)

from OgreInterface.interfaces import Interface
from OgreInterface import utils


class CHGNetSurfaceMatcher(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        chgnet_model: tp.Optional[str] = None,
        grid_density: float = 2.5,
    ):
        try:
            from chgnet.model.model import CHGNet
        except ImportError:
            raise "You need to install `chgnet` in order to use the ChgNetSurfaceMatcher"

        super().__init__(interface=interface, grid_density=grid_density)

        # TODO: Implement chgnet
        self.chgnet_model = chgnet_model

        if self.chgnet_model is None:
            self.model = CHGNet.load()
        else:
            self.model = CHGNet.from_file(self.chgnet_model)

        self.surface_energy_kwargs = {"chgnet_model": self.chgnet_model}

    def generate_constant_inputs(
        self,
        structure: Structure,
    ) -> tp.List[Structure]:
        """
        This method is used to generate the inputs of the calculate function
        for the structures that will stay constant throughout the surface
        matching process (i.e. OBS, supercells)
        """
        return [structure]

    def generate_interface_inputs(
        self,
        shifts: np.ndarray,
    ) -> tp.List[Structure]:
        """
        This method is used to generate the inputs of the calculate function
        for the interface given various shifts
        """
        shifted_ifaces = []
        for shift in shifts:
            shifted_iface = utils.shift_film(
                interface=self.iface,
                shift=shift,
                fractional=False,
            )
            shifted_ifaces.append(shifted_iface)

        return shifted_ifaces

    def calculate(self, inputs: tp.List[Structure]) -> np.array:
        """
        This method is used to calculate the total energy of the structure with
        the given method of calculating the energy (i.e. DFT, ML-potential)
        """
        model_outputs = self.model.predict_structure(
            structure=inputs,
            task="e",
        )

        n_atoms = np.array([len(s) for s in inputs])

        if type(model_outputs) is dict:
            energies = np.array([model_outputs["e"]])
        else:
            energies = np.array([i["e"] for i in model_outputs])

        return energies * n_atoms

    @property
    def surface_energy_module(self) -> CHGNetSurfaceEnergy:
        """
        Set the surface energy module here
        """
        return CHGNetSurfaceEnergy
