import typing as tp

from pymatgen.core.structure import Structure
import numpy as np

from OgreInterface.surface_matching import (
    BaseSurfaceEnergy,
)
from OgreInterface.surfaces import Surface
from OgreInterface import utils


class CHGNetSurfaceEnergy(BaseSurfaceEnergy):
    def __init__(
        self,
        surface: Surface,
        chgnet_model: tp.Optional[str] = None,
    ):
        try:
            from chgnet.model.model import CHGNet
        except ImportError:
            raise "You need to install `chgnet` in order to use ChgNetSurfaceEnergy"

        super().__init__(surface=surface)

        # TODO: Implement chgnet
        self.chgnet_model = chgnet_model

        if self.chgnet_model is None:
            self.model = CHGNet.load()
        else:
            self.model = CHGNet.from_file(self.chgnet_model)

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
                interface=self.double_slab,
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
