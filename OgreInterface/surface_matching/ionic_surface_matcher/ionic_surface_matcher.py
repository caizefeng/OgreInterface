from typing import List, Dict, Tuple, Optional
import tpying as tp
import itertools

# from itertools import groupby, combinations_with_replacement, product
from os.path import join, dirname, split, abspath

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import chemical_symbols, covalent_radii
from matscipy.neighbours import neighbour_list
import pandas as pd
import numpy as np

from OgreInterface import data
from OgreInterface.surface_matching.ionic_surface_matcher.input_generator import (
    generate_input_dict,
    create_batch,
)
from OgreInterface.surface_matching.ionic_surface_matcher.ionic_shifted_force_potential import (
    IonicShiftedForcePotential,
)
import OgreInterface.surface_matching.ionic_surface_matcher.utils as ionic_utils
from OgreInterface.surfaces import Interface
from OgreInterface.surface_matching.base_surface_matcher import (
    BaseSurfaceMatcher,
)


DATA_PATH = dirname(abspath(data.__file__))


class IonicSurfaceMatcher(BaseSurfaceMatcher):
    """Class to perform surface matching between ionic materials

    The IonicSurfaceMatcher class contain various methods to perform surface matching
    specifically tailored towards an interface between two ionic materials.

    Examples:
        Calculating the 2D potential energy surface (PES)
        >>> from OgreInterface.surface_match import IonicSurfaceMatcher
        >>> surface_matcher = IonicSurfaceMatcher(interface=interface) # interface is Interface class
        >>> E_opt = surface_matcher.run_surface_matching(output="PES.png")
        >>> surface_matcher.get_optmized_structure() # Shift the interface to it's optimal position

        Optimizing the interface in 3D using particle swarm optimization
        >>> from OgreInterface.surface_match import IonicSurfaceMatcher
        >>> surface_matcher = IonicSurfaceMatcher(interface=interface) # interface is Interface class
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
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )

        # Set PBC for the surfaces so the z-direction is False
        # so the neighbor finding algo doesn't search z-images
        self.iface.lattice._pbc = (True, True, False)
        self.film_supercell.lattice._pbc = (True, True, False)
        self.sub_supercell.lattice._pbc = (True, True, False)

        # Get the ionic radii data from the data module
        self._ionic_radii_df = pd.read_csv(
            join(DATA_PATH, "ionic_radii_data.csv")
        )

        # Determined if the born n should be set based on the
        # electron configuration (I think this should just be False)
        self._auto_determine_born_n = auto_determine_born_n

        # Manual born n value. (born_n = 12 usually gives best results)
        self._born_n = born_n

        # Cutoff for neighbor finding
        self._cutoff = 18.0

        # Get charge dictionary mapping
        # {"sub":{chemical_symbol: charge}, "film":{chemical_symbols: charge}}
        self.charge_dict = self._get_charges()

        # Get bulk equivalent to atomic number mapping
        # {"sub":{bulk_eq: atomic_number}, "film":{bulk_eq: atomic_number}}
        self.eq_to_Z_dict = self._get_equiv_to_Zs()

        # Get bulk equivalent to ionic radius mapping
        # {"sub":{bulk_eq: radius}, "film":{bulk_eq: radius}}
        self.r0_dict = self._get_r0s()

        # Get max z when searching using PSO
        self._max_z = self._get_max_z()

        self.surface_energy_module = IonicSurfaceEnergy
        self.surface_energy_kwargs = {
            "film": {
                "oriented_bulk_structure": self.interface.film_oriented_bulk_structure,
                "layers": self.interface.film.layers,
                "r0_dict": self.r0_dict["film"],
                "charge_dict": self.charge_dict["film"],
            },
            "sub": {
                "oriented_bulk_structure": self.interface.substrate_oriented_bulk_structure,
                "layers": self.interface.substrate.layers,
                "r0_dict": self.r0_dict["sub"],
                "charge_dict": self.charge_dict["sub"],
            },
        }

        # Add born ns to all the structures
        self._add_born_ns(self.iface)
        self._add_born_ns(self.sub_sc_part)
        self._add_born_ns(self.film_sc_part)

        # Add r0s to all the structures
        self._add_r0s(self.iface)
        self._add_r0s(self.sub_sc_part)
        self._add_r0s(self.film_sc_part)

        # Set the interfacial distance
        self.d_interface = self.interface.interfacial_distance

        # Placeholder for the optimal xy shift
        self.opt_xy_shift = np.zeros(2)

        # Placeholder for the optimal interfacial distance
        self.opt_d_interface = self.d_interface

        # Generate the base inputs for the interface
        all_iface_inputs = ionic_utils.generate_base_inputs(
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

        # Generate the base inputs for the substrate supercell
        self.sub_sc_inputs = self._generate_base_inputs(
            structure=self.sub_sc_part,
            cutoff=self._cutoff + 5.0,
        )

        # Generate the base inputs for the film supercell
        self.film_sc_inputs = self._generate_base_inputs(
            structure=self.film_sc_part,
            cutoff=self._cutoff + 5.0,
        )

        (
            self.film_energy,
            self.sub_energy,
            self.film_surface_energy,
            self.sub_surface_energy,
            self.const_iface_energy,
        ) = self._get_const_energies()

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_charges = ionic_utils.get_charges_from_structure(structure=sub)
        film_charges = ionic_utils.get_charges_from_structure(structure=film)

        return {"sub": sub_charges, "film": film_charges}

    def _get_equiv_to_Zs(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_equiv_to_Z = (
            ionic_utils.get_equivalent_site_to_atomic_number_mapping(
                structure=sub
            )
        )
        film_equiv_to_Z = (
            ionic_utils.get_equivalent_site_to_atomic_number_mapping(
                structure=film
            )
        )

        return {"sub": sub_equiv_to_Z, "film": film_equiv_to_Z}

    def _get_r0s(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        sub_radii_dict = ionic_utils.get_ionic_radii_from_structure(
            structure=sub,
            charge_dict=self.charge_dict["sub"],
            equiv_to_Z_dict=self.equiv_to_Z_dict["sub"],
        )

        film_radii_dict = ionic_utils.get_ionic_radii_from_structure(
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
        eq_to_Z = self.eq_to_Z_dict

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

    def _add_r0s(self, struc):
        r0s = []

        for site in struc:
            atomic_number = site.properties["bulk_equivalent"]
            if bool(site.properties["is_film"]):
                r0s.append(self.r0_dict["film"][atomic_number])
            else:
                r0s.append(self.r0_dict["sub"][atomic_number])

        struc.add_site_property("r0s", r0s)

    def _add_born_ns(self, struc):
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
            inputs=self._iface_inputs,
            batch_size=len(shifts),
        )

        self._add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shifts,
        )

        batch_inputs["is_interface"] = True

        return batch_inputs

    def get_optimized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )
        self.interface.set_interfacial_distance(
            interfacial_distance=self.opt_d_interface
        )

        self.iface = self.interface.get_interface(orthogonal=True).copy()

        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        self._add_born_ns(self.iface)
        self._add_r0s(self.iface)
        iface_inputs = self._generate_base_inputs(
            structure=self.iface,
            is_slab=True,
        )
        _, self.iface_inputs = self._get_constant_and_variable_iface_inputs(
            inputs=iface_inputs
        )

        self.opt_xy_shift[:2] = 0.0
        self.d_interface = self.opt_d_interface

    def pso_function(self, x):
        cart_xy = self.get_cart_xy_shifts(x[:, :2])
        z_shift = x[:, -1] - self.d_interface
        shift = np.c_[cart_xy, z_shift]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=len(x),
        )

        self._add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shift,
        )

        E, _, _, _, _ = self._calculate(
            inputs=batch_inputs,
            is_interface=True,
        )
        E_adh, E_iface = self._get_interface_energy(total_energies=E)

        return E_iface

    def optimizePSO(
        self,
        z_bounds: Optional[List[float]] = None,
        max_iters: int = 200,
        n_particles: int = 15,
    ) -> float:
        """
        This function will optimize the interface structure in 3D using Particle Swarm Optimization

        Args:
            z_bounds: A list defining the maximum and minumum interfacial distance [min, max]
            max_iters: Maximum number of iterations of the PSO algorithm
            n_particles: Number of particles to use for the swarm (10 - 20 is usually sufficient)

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        if z_bounds is None:
            z_bounds = [0.5, max(3.5, 1.2 * self._max_z)]

        opt_score, opt_pos = self._optimizerPSO(
            func=self.pso_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            n_particles=n_particles,
        )

        opt_cart = self.get_cart_xy_shifts(opt_pos[:2].reshape(1, -1))
        opt_cart = np.c_[opt_cart, np.zeros(1)]
        opt_frac = opt_cart.dot(self.inv_matrix)[0]

        self.opt_xy_shift = opt_frac[:2]
        self.opt_d_interface = opt_pos[-1]

        return opt_score

    def _get_const_energies(self):
        sub_sc_inputs = create_batch(
            inputs=self.sub_sc_inputs,
            batch_size=1,
        )
        film_sc_inputs = create_batch(
            inputs=self.film_sc_inputs,
            batch_size=1,
        )

        const_iface_inputs = create_batch(
            inputs=self.const_iface_inputs,
            batch_size=1,
        )

        sub_sc_energy, _, _, _, _ = self._calculate(
            sub_sc_inputs,
            is_interface=False,
        )
        film_sc_energy, _, _, _, _ = self._calculate(
            film_sc_inputs,
            is_interface=False,
        )

        (
            const_iface_energy,
            _,
            self.const_born_energy,
            self.const_coulomb_energy,
            _,
        ) = self._calculate(
            const_iface_inputs,
            is_interface=False,
        )

        film_surface_energy_calc = self.surface_energy_module(
            **self.surface_energy_kwargs["film"]
        )
        film_surface_energy = film_surface_energy_calc.get_cleavage_energy()

        sub_surface_energy_calc = self.surface_energy_module(
            **self.surface_energy_kwargs["sub"]
        )
        sub_surface_energy = sub_surface_energy_calc.get_cleavage_energy()

        return (
            film_sc_energy[0],
            sub_sc_energy[0],
            film_surface_energy,
            sub_surface_energy,
            const_iface_energy[0],
        )

    def calculate(self, inputs: Dict):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )

        if inputs["is_interface"]:
            energy = ionic_potential.forward(
                inputs=inputs,
                constant_coulomb_contribution=self.const_coulomb_energy,
                constant_born_contribution=self.const_born_energy,
            )
        else:
            energy = ionic_potential.forward(
                inputs=inputs,
            )

        return energy
