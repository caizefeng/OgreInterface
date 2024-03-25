import typing as tp

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from ase.data import chemical_symbols
import numpy as np

from OgreInterface.surface_matching import (
    BaseSurfaceEnergy,
)
from OgreInterface.surface_matching.ionic_surface_matcher import (
    generate_input_dict,
    create_batch,
    IonicShiftedForcePotential,
    ionic_utils,
)
from OgreInterface.surfaces import Surface


class IonicSurfaceEnergy(BaseSurfaceEnergy):
    def __init__(
        self,
        surface: Surface,
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
    ):
        # Cutoff for neighbor finding
        self._cutoff = 18.0

        super().__init__(surface=surface)

        # Set PBC for the surfaces so the z-direction is False
        # so the neighbor finding algo doesn't search z-images
        self.slab.lattice._pbc = (True, True, False)
        self.double_slab.lattice._pbc = (True, True, False)

        # Determined if the born n should be set based on the
        # electron configuration (I think this should just be False)
        self._auto_determine_born_n = auto_determine_born_n

        # Manual born n value. (born_n = 12 usually gives best results)
        self._born_n = born_n

        # Get charge dictionary mapping
        # {"sub":{chemical_symbol: charge}, "film":{chemical_symbols: charge}}
        self.charge_dict = self._get_charges()

        # Get bulk equivalent to atomic number mapping
        # {"sub":{bulk_eq: atomic_number}, "film":{bulk_eq: atomic_number}}
        self.equiv_to_Z_dict = self._get_equiv_to_Zs()

        # Get bulk equivalent to ionic radius mapping
        # {"sub":{bulk_eq: radius}, "film":{bulk_eq: radius}}
        self.r0_dict = self._get_r0s()

        # Add born ns to all the structures
        self._set_born_ns(self.slab)
        self._set_born_ns(self.double_slab)
        self._set_born_ns(self.obs)

        # Add r0s to all the structures
        self._set_r0s(self.slab)
        self._set_r0s(self.double_slab)
        self._set_r0s(self.obs)

        # Generate the base inputs for the interface
        all_double_slab_inputs = ionic_utils.generate_base_inputs(
            structure=self.double_slab,
            cutoff=self._cutoff + 5.0,
        )

        # Split the interface inputs into
        # (film-film & sub-sub, and film-sub interactions)
        (
            self.const_double_slab_inputs,
            self.double_slab_inputs,
        ) = self._get_constant_and_variable_iface_inputs(
            inputs=all_double_slab_inputs
        )

        (
            self.const_born_energy,
            self.const_coulomb_energy,
        ) = self._get_constant_interface_terms()

    def generate_constant_inputs(
        self,
        structure: Structure,
    ) -> tp.Dict[str, np.ndarray]:
        inputs = generate_input_dict(
            structure=structure,
            cutoff=self._cutoff + 5.0,
        )

        batch_inputs = create_batch(
            inputs=inputs,
            batch_size=1,
        )

        batch_inputs["is_interface"] = False

        return batch_inputs

    def generate_interface_inputs(
        self,
        shifts: np.ndarray,
    ) -> tp.Dict[str, np.ndarray]:
        batch_inputs = create_batch(
            inputs=self.double_slab_inputs,
            batch_size=len(shifts),
        )

        ionic_utils.add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shifts,
        )

        batch_inputs["is_interface"] = True

        return batch_inputs

    def calculate(
        self,
        inputs: tp.Dict[str, tp.Union[np.ndarray, bool]],
    ) -> np.ndarray:
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )

        if inputs["is_interface"]:
            (
                energy,
                _,
                _,
                _,
            ) = ionic_potential.forward(
                inputs=inputs,
                constant_coulomb_contribution=self.const_coulomb_energy,
                constant_born_contribution=self.const_born_energy,
            )
        else:
            (
                energy,
                _,
                _,
                _,
            ) = ionic_potential.forward(inputs=inputs)

        return energy

    def _get_charges(self):
        bulk = self.surface.bulk_structure

        charge_dict = ionic_utils.get_charges_from_structure(structure=bulk)

        return charge_dict

    def _get_equiv_to_Zs(self):
        bulk = self.surface.bulk_structure

        equiv_to_Z_dict = (
            ionic_utils.get_equivalent_site_to_atomic_number_mapping(
                structure=bulk
            )
        )

        return equiv_to_Z_dict

    def _get_r0s(self):
        bulk = self.surface.bulk_structure

        r0_dict = ionic_utils.get_ionic_radii_from_structure(
            structure=bulk,
            charge_dict=self.charge_dict,
            equiv_to_Z_dict=self.equiv_to_Z_dict,
        )

        return r0_dict

    def _get_constant_and_variable_iface_inputs(
        self,
        inputs: tp.Dict[str, np.ndarray],
    ) -> tp.Tuple[tp.Dict[str, np.ndarray], tp.Dict[str, np.ndarray]]:
        film_film_mask = (
            inputs["is_film"][inputs["idx_i"]]
            & inputs["is_film"][inputs["idx_j"]]
        )
        sub_sub_mask = (~inputs["is_film"])[inputs["idx_i"]] & (
            ~inputs["is_film"]
        )[inputs["idx_j"]]

        const_mask = np.logical_or(film_film_mask, sub_sub_mask)

        const_inputs = {}
        variable_inputs = {}

        for k, v in inputs.items():
            if "idx" in k or "offsets" in k:
                const_inputs[k] = v[const_mask]
                variable_inputs[k] = v[~const_mask]
            else:
                const_inputs[k] = v
                variable_inputs[k] = v

        return const_inputs, variable_inputs

    def _get_constant_interface_terms(self):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )

        const_iface_inputs = create_batch(
            inputs=self.const_double_slab_inputs,
            batch_size=1,
        )

        (
            _,
            _,
            constant_born,
            constant_coulomb,
        ) = ionic_potential.forward(inputs=const_iface_inputs)

        return constant_born, constant_coulomb

    def _set_r0s(self, struc):
        r0s = []

        for site in struc:
            atomic_number = site.properties["bulk_equivalent"]
            r0s.append(self.r0_dict[atomic_number])

        struc.add_site_property("r0s", r0s)

    def _set_born_ns(self, struc):
        ion_config_to_n_map = {
            "1s1": 0.0,
            "[He]": 5.0,
            "[Ne]": 7.0,
            "[Ar]": 9.0,
            "[Kr]": 10.0,
            "[Xe]": 12.0,
        }
        n_vals = {}

        Zs = np.unique(struc.atomic_numbers)
        for z in Zs:
            element = Element(chemical_symbols[z])
            ion_config = element.electronic_structure.split(".")[0]
            n_val = ion_config_to_n_map[ion_config]
            if self._auto_determine_born_n:
                n_vals[z] = n_val
            else:
                n_vals[z] = self._born_n

        ns = [n_vals[z] for z in struc.atomic_numbers]
        struc.add_site_property("born_ns", ns)
