import typing as tp

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from ase.data import chemical_symbols
import numpy as np

from OgreInterface.surface_matching import (
    BaseSurfaceEnergy,
)

from OgreInterface.surface_matching.lj_surface_matcher import (
    generate_input_dict,
    create_batch,
    LJPotential,
    lj_utils,
)

from OgreInterface.surfaces import Surface


class LJSurfaceEnergy(BaseSurfaceEnergy):
    def __init__(
        self,
        surface: Surface,
    ):
        # Cutoff for neighbor finding
        self._cutoff = 12.0

        super().__init__(surface=surface)

        # Set PBC for the surfaces so the z-direction is False
        # so the neighbor finding algo doesn't search z-images
        self.slab.lattice._pbc = (True, True, False)
        self.double_slab.lattice._pbc = (True, True, False)

        # Get charge dictionary mapping
        # {"sub":{chemical_symbol: charge}, "film":{chemical_symbols: charge}}
        self.charge_dict = self._get_charges()

        # Get bulk equivalent to atomic number mapping
        # {"sub":{bulk_eq: atomic_number}, "film":{bulk_eq: atomic_number}}
        self.equiv_to_Z_dict = self._get_equiv_to_Zs()

        # Get bulk equivalent to ionic radius mapping
        # {"sub":{bulk_eq: radius}, "film":{bulk_eq: radius}}
        self.r0_dict = self._get_r0s()

        # Add r0s to all the structures
        self._set_r0s(self.slab)
        self._set_r0s(self.double_slab)
        self._set_r0s(self.obs)

        # Generate the base inputs for the interface
        all_double_slab_inputs = lj_utils.generate_base_inputs(
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

        self.const_energy = self._get_constant_interface_terms()

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

        lj_utils.add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shifts,
        )

        batch_inputs["is_interface"] = True

        return batch_inputs

    def calculate(
        self,
        inputs: tp.Dict[str, tp.Union[np.ndarray, bool]],
    ) -> np.ndarray:
        potential = LJPotential(
            cutoff=self._cutoff,
        )

        energy = potential.forward(inputs=inputs)

        if inputs["is_interface"]:
            energy += self.const_energy

        return energy

    def _get_charges(self):
        bulk = self.surface.bulk_structure

        charge_dict = lj_utils.get_charges_from_structure(structure=bulk)

        return charge_dict

    def _get_equiv_to_Zs(self):
        bulk = self.surface.bulk_structure

        equiv_to_Z_dict = (
            lj_utils.get_equivalent_site_to_atomic_number_mapping(
                structure=bulk
            )
        )

        return equiv_to_Z_dict

    def _get_r0s(self):
        bulk = self.surface.bulk_structure

        r0_dict = lj_utils.get_radii_from_structure(
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
        potential = LJPotential(
            cutoff=self._cutoff,
        )

        const_iface_inputs = create_batch(
            inputs=self.const_double_slab_inputs,
            batch_size=1,
        )

        const_energy = potential.forward(inputs=const_iface_inputs)

        return const_energy

    def _set_r0s(self, struc):
        r0s = []

        for site in struc:
            atomic_number = site.properties["bulk_equivalent"]
            r0s.append(self.r0_dict[atomic_number])

        struc.add_site_property("r0s", r0s)
