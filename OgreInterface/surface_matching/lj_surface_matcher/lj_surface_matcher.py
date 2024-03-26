import typing as tp
import itertools

from pymatgen.core.structure import Structure
from ase.data import chemical_symbols
import numpy as np

from OgreInterface.surface_matching.lj_surface_matcher import (
    generate_input_dict,
    create_batch,
    LJPotential,
    lj_utils,
)

from OgreInterface.surface_matching import (
    LJSurfaceEnergy,
    BaseSurfaceMatcher,
)

from OgreInterface.interfaces import Interface


class LJSurfaceMatcher(BaseSurfaceMatcher):
    """Class to perform surface matching between ionic materials

    The LJSurfaceMatcher class contain various methods to perform surface matching
    specifically tailored towards an interface between two ionic materials.

    Examples:
        Calculating the 2D potential energy surface (PES)
        >>> from OgreInterface.surface_match import LJSurfaceMatcher
        >>> surface_matcher = LJSurfaceMatcher(interface=interface) # interface is Interface class
        >>> E_opt = surface_matcher.run_surface_matching(output="PES.png")
        >>> surface_matcher.get_optmized_structure() # Shift the interface to it's optimal position

        Optimizing the interface in 3D using particle swarm optimization
        >>> from OgreInterface.surface_match import LJSurfaceMatcher
        >>> surface_matcher = LJSurfaceMatcher(interface=interface) # interface is Interface class
        >>> E_opt = surface_matcher.optimizePSO(z_bounds=[1.0, 5.0], max_iters=150, n_particles=12)
        >>> surface_matcher.get_optmized_structure() # Shift the interface to it's optimal position

    Args:
        interface: The Interface object generated using the InterfaceGenerator
        grid_density: The sampling density of the 2D potential energy surface plot (points/Angstrom)
    """

    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
        verbose: bool = True,
    ):
        # Cutoff for neighbor finding
        self._cutoff = 12.0

        super().__init__(
            interface=interface,
            grid_density=grid_density,
            verbose=verbose,
        )

        # Set PBC for the surfaces so the z-direction is False
        # so the neighbor finding algo doesn't search z-images
        self.iface.lattice._pbc = (True, True, False)
        self.film_supercell.lattice._pbc = (True, True, False)
        self.sub_supercell.lattice._pbc = (True, True, False)

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
        self._set_r0s(self.iface)
        self._set_r0s(self.sub_supercell)
        self._set_r0s(self.film_supercell)

        # Generate the base inputs for the interface
        all_iface_inputs = lj_utils.generate_base_inputs(
            structure=self.iface,
            cutoff=self._cutoff + 5.0,
        )

        # Split the interface inputs into
        # (film-film & sub-sub, and film-sub interactions)
        (
            self.const_iface_inputs,
            self.iface_inputs,
        ) = self._get_constant_and_variable_iface_inputs(
            inputs=all_iface_inputs
        )

        self.const_energy = self._get_constant_interface_terms()

        self.surface_energy_kwargs = {}

    @property
    def surface_energy_module(self) -> LJSurfaceEnergy:
        return LJSurfaceEnergy

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
            inputs=self.iface_inputs,
            batch_size=len(shifts),
        )

        lj_utils.add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shifts,
        )

        batch_inputs["is_interface"] = True

        return batch_inputs

    def get_optimized_structure(self):
        # Shift the interface to the optimal inplane positon
        self.interface.shift_film_inplane(
            x_shift=self.opt_xy_shift[0],
            y_shift=self.opt_xy_shift[1],
            fractional=True,
        )

        # Set the interfacial distance to the optimal interfacial distance
        self.interface.set_interfacial_distance(
            interfacial_distance=self.opt_d_interface
        )

        # Reset the self.iface property
        self.iface = self.interface.get_interface(orthogonal=True).copy()

        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        # Add the r0s to the iface structure
        self._set_r0s(self.iface)

        # Generate the base inputs for the interface
        all_iface_inputs = lj_utils.generate_base_inputs(
            structure=self.iface,
            cutoff=self._cutoff + 5.0,
        )

        # Recalculate the variable iface inputs
        (
            _,
            self.iface_inputs,
        ) = self._get_constant_and_variable_iface_inputs(
            inputs=all_iface_inputs
        )

        # Reset optimal shift values
        self.opt_xy_shift[:2] = 0.0
        self.d_interface = self.opt_d_interface

    def calculate(
        self,
        inputs: tp.Dict[str, tp.Union[np.ndarray, bool]],
    ) -> np.ndarray:
        ionic_potential = LJPotential(
            cutoff=self._cutoff,
        )

        energy = ionic_potential.forward(inputs=inputs)

        if inputs["is_interface"]:
            energy += self.const_energy

        return energy

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_charges = lj_utils.get_charges_from_structure(structure=sub)
        film_charges = lj_utils.get_charges_from_structure(structure=film)

        return {"sub": sub_charges, "film": film_charges}

    def _get_equiv_to_Zs(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_equiv_to_Z = lj_utils.get_equivalent_site_to_atomic_number_mapping(
            structure=sub
        )
        film_equiv_to_Z = (
            lj_utils.get_equivalent_site_to_atomic_number_mapping(
                structure=film
            )
        )

        return {"sub": sub_equiv_to_Z, "film": film_equiv_to_Z}

    def _get_r0s(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_radii_dict = lj_utils.get_radii_from_structure(
            structure=sub,
            charge_dict=self.charge_dict["sub"],
            equiv_to_Z_dict=self.equiv_to_Z_dict["sub"],
        )

        film_radii_dict = lj_utils.get_radii_from_structure(
            structure=film,
            charge_dict=self.charge_dict["film"],
            equiv_to_Z_dict=self.equiv_to_Z_dict["film"],
        )

        r0_dict = {"film": film_radii_dict, "sub": sub_radii_dict}
        # eq_to_Z_dict = {"film": film_eq_to_Z_dict, "sub": sub_eq_to_Z_dict}

        return r0_dict

    def _get_max_z(self) -> float:
        charges = self.charge_dict
        r0s = self.r0_dict
        eq_to_Z = self.equiv_to_Z_dict

        positive_r0s = []
        negative_r0s = []

        for sub_film, sub_film_r0s in r0s.items():
            for eq, r in sub_film_r0s.items():
                chg = charges[sub_film][
                    chemical_symbols[eq_to_Z[sub_film][eq]]
                ]
                if np.sign(chg) > 0:
                    positive_r0s.append(r)
                elif np.sign(chg) < 0:
                    negative_r0s.append(r)
                else:
                    positive_r0s.append(r)
                    negative_r0s.append(r)

        combos = itertools.product(positive_r0s, negative_r0s)
        bond_lengths = [sum(combo) for combo in combos]
        max_bond_length = max(bond_lengths)

        return max_bond_length

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

    def _set_r0s(self, struc):
        r0s = []

        for site in struc:
            atomic_number = site.properties["bulk_equivalent"]
            if bool(site.properties["is_film"]):
                r0s.append(self.r0_dict["film"][atomic_number])
            else:
                r0s.append(self.r0_dict["sub"][atomic_number])

        struc.add_site_property("r0s", r0s)

    def _get_constant_interface_terms(self):
        potential = LJPotential(
            cutoff=self._cutoff,
        )

        const_iface_inputs = create_batch(
            inputs=self.const_iface_inputs,
            batch_size=1,
        )

        const_energy = potential.forward(inputs=const_iface_inputs)

        return const_energy
