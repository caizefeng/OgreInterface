import typing as tp

from pymatgen.core.structure import Structure

from OgreInterface.energy_modules.base_energy_module import BaseEnergyModule


class IonicEnergyModule(BaseEnergyModule):
    def __init__(self):
        super().__init__()

    def preprocess(
        self,
        bulk_structure_dict: tp.Dict[str, Structure],
    ):
        """
        This will perform the preprocessing required to parameterize the
        ionic potential.

        Args:
            bulk_structure_dict: Dictionary of bulk structure of the form
                {"sub": Structure, "film": Structure}

        """
        pass
