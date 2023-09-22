from typing import Union, List, Tuple, Optional
from os.path import isdir, join
import os

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from OgreInterface.generate import InterfaceGenerator, SurfaceGenerator
from OgreInterface.surface_match import IonicSurfaceMatcher
from OgreInterface.surface_energy import IonicSurfaceEnergy
from OgreInterface.plotting_tools import plot_surface_charge_matrix
from OgreInterface import utils


class IonicInterfaceSearch:
    """Class to perform a miller index scan to find all domain matched interfaces of various surfaces.

    Examples:
        >>> from OgreInterface.miller import MillerSearch
        >>> ms = MillerSearch(substrate="POSCAR_sub", film="POSCAR_film", max_substrate_index=1, max_film_index=1)
        >>> ms.run_scan()
        >>> ms.plot_misfits(output="miller_scan.png")

    Args:
        substrate: Bulk structure of the substrate in either Pymatgen Structure, ASE Atoms, or a structure file such as a POSCAR or Cif
        film: Bulk structure of the film in either Pymatgen Structure, ASE Atoms, or a structure file such as a POSCAR or Cif
        max_substrate_index: Max miller index of the substrate surfaces
        max_film_index: Max miller index of the film surfaces
        minimum_slab_thickness: Determines the minimum thickness of the film and substrate slabs
        max_area_mismatch: Area ratio mismatch tolerance for the InterfaceGenerator
        max_angle_strain: Angle strain tolerance for the InterfaceGenerator
        max_linear_strain: Lattice vectors length mismatch tolerance for the InterfaceGenerator
        max_area: Maximum area of the matched supercells
        refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
            This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
            users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
            work exclusively with the primitive structure so we always have it on hand.
    """

    def __init__(
        self,
        substrate_bulk: Union[Structure, Atoms, str],
        film_bulk: Union[Structure, Atoms, str],
        substrate_miller_index: List[int],
        film_miller_index: List[int],
        minimum_slab_thickness: float = 18.0,
        max_area_mismatch: float = 0.01,
        max_strain: float = 0.01,
        max_area: Optional[float] = None,
        substrate_strain_fraction: float = 0.0,
        refine_structure: bool = True,
        suppress_warnings: bool = True,
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
        n_particles_PSO: int = 20,
        max_iterations_PSO: int = 150,
        z_bounds_PSO: List[float] = [1.0, 6.0],
        grid_density_PES: float = 2.5,
        use_most_stable_substrate: bool = True,
        cmap_PES="coolwarm",
    ):
        self._refine_structure = refine_structure
        self._suppress_warnings = suppress_warnings
        if type(substrate_bulk) == str:
            self._substrate_bulk, _ = self._get_bulk(
                Structure.from_file(substrate_bulk)
            )
        else:
            self._substrate_bulk, _ = self._get_bulk(substrate_bulk)

        if type(film_bulk) == str:
            self._film_bulk, _ = self._get_bulk(Structure.from_file(film_bulk))
        else:
            self._film_bulk, _ = self._get_bulk(film_bulk)

        self._auto_determine_born_n = auto_determine_born_n
        self._born_n = born_n
        self._n_particles_PSO = n_particles_PSO
        self._max_iterations_PSO = max_iterations_PSO
        self._z_bounds_PSO = z_bounds_PSO
        self._use_most_stable_substrate = use_most_stable_substrate
        self._grid_density_PES = grid_density_PES
        self._minimum_slab_thickness = minimum_slab_thickness
        self._substrate_miller_index = substrate_miller_index
        self._film_miller_index = film_miller_index
        self._max_area_mismatch = max_area_mismatch
        self._max_strain = max_strain
        self._substrate_strain_fraction = substrate_strain_fraction
        self._max_area = max_area
        self._cmap_PES = cmap_PES

    def _get_bulk(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
        else:
            raise TypeError(
                f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        if self._refine_structure:
            conventional_structure = utils.spglib_standardize(
                init_structure,
                to_primitive=False,
                no_idealize=False,
            )

            init_angles = init_structure.lattice.angles
            init_lengths = init_structure.lattice.lengths
            init_length_and_angles = np.concatenate(
                [list(init_lengths), list(init_angles)]
            )

            conv_angles = conventional_structure.lattice.angles
            conv_lengths = conventional_structure.lattice.lengths
            conv_length_and_angles = np.concatenate(
                [list(conv_lengths), list(conv_angles)]
            )

            if not np.isclose(
                conv_length_and_angles - init_length_and_angles, 0
            ).all():
                if not self._suppress_warnings:
                    labels = ["a", "b", "c", "alpha", "beta", "gamma"]
                    init_cell_str = ", ".join(
                        [
                            f"{label} = {val:.3f}"
                            for label, val in zip(
                                labels, init_length_and_angles
                            )
                        ]
                    )
                    conv_cell_str = ", ".join(
                        [
                            f"{label} = {val:.3f}"
                            for label, val in zip(
                                labels, conv_length_and_angles
                            )
                        ]
                    )
                    warning_str = "\n".join(
                        [
                            "----------------------------------------------------------",
                            "WARNING: The refined cell is different from the input cell",
                            f"Initial: {init_cell_str}",
                            f"Refined: {conv_cell_str}",
                            "Make sure the input miller index is for the refined structure, otherwise set refine_structure=False",
                            "To turn off this warning set suppress_warnings=True",
                            "----------------------------------------------------------",
                            "",
                        ]
                    )
                    print(warning_str)

            conventional_atoms = AseAtomsAdaptor.get_atoms(
                conventional_structure
            )

            return (
                conventional_structure,
                conventional_atoms,
            )
        else:
            init_atoms = AseAtomsAdaptor().get_atoms(init_structure)

            return init_structure, init_atoms

    def _get_surface_generators(self):
        substrate_generator = SurfaceGenerator(
            bulk=self._substrate_bulk,
            miller_index=self._substrate_miller_index,
            layers=None,
            minimum_thickness=self._minimum_slab_thickness,
            vacuum=10,
            refine_structure=self._refine_structure,
        )

        film_generator = SurfaceGenerator(
            bulk=self._film_bulk,
            miller_index=self._film_miller_index,
            layers=None,
            minimum_thickness=self._minimum_slab_thickness,
            vacuum=10,
            refine_structure=True,
        )

        return substrate_generator, film_generator

    def _get_most_stable_surface(
        self, surface_generator: SurfaceGenerator
    ) -> List[int]:
        surface_energies = []
        for surface in surface_generator:
            surfE_calculator = IonicSurfaceEnergy(surface=surface)
            surface_energies.append(surfE_calculator.get_surface_energy())

        surface_energies = np.round(np.array(surface_energies), 6)
        min_surface_energy = surface_energies.min()

        most_stable_indices = np.where(surface_energies == min_surface_energy)

        return most_stable_indices[0]

    def _get_film_and_substrate_inds(
        self,
        film_generator: SurfaceGenerator,
        substrate_generator: SurfaceGenerator,
        filter_on_charge: bool = True,
    ) -> List[Tuple[int, int]]:
        film_and_substrate_inds = []

        if self._use_most_stable_substrate:
            substrate_inds_to_use = self._get_most_stable_surface(
                surface_generator=substrate_generator
            )
        else:
            substrate_inds_to_use = np.arange(len(substrate_generator)).astype(
                int
            )

        for i, film in enumerate(film_generator):
            for j, sub in enumerate(substrate_generator):
                if j in substrate_inds_to_use:
                    if filter_on_charge:
                        sub_sign = np.sign(sub.top_surface_charge)
                        film_sign = np.sign(film.bottom_surface_charge)

                        if sub_sign == 0.0 or film_sign == 0.0:
                            film_and_substrate_inds.append((i, j))
                        else:
                            if np.sign(sub_sign * film_sign) < 0.0:
                                film_and_substrate_inds.append((i, j))
                    else:
                        film_and_substrate_inds.append((i, j))

        return film_and_substrate_inds

    def run_interface_search(
        self,
        filter_on_charge: bool = True,
        output_folder: str = None,
    ):
        if output_folder is None:
            sub_comp = self._substrate_bulk.composition.reduced_formula
            film_comp = self._film_bulk.composition.reduced_formula
            sub_miller = "".join(
                [str(i) for i in self._substrate_miller_index]
            )
            film_miller = "".join([str(i) for i in self._film_miller_index])
            base_dir = f"{sub_comp}{sub_miller}_{film_comp}{film_miller}"

            current_dirs = [d for d in os.listdir() if base_dir in d]

            if len(current_dirs) > 0:
                base_dir += f"_{len(current_dirs)}"
        else:
            base_dir = output_folder

        if not isdir(base_dir):
            os.mkdir(base_dir)

        substrate_generator, film_generator = self._get_surface_generators()

        plot_surface_charge_matrix(
            films=film_generator,
            substrates=substrate_generator,
            output=join(base_dir, "surface_charge_matrix.png"),
        )

        film_and_substrate_inds = self._get_film_and_substrate_inds(
            film_generator=film_generator,
            substrate_generator=substrate_generator,
            filter_on_charge=filter_on_charge,
        )

        energies = []
        opt_abz_shifts = []
        surface_charges = []
        surface_energies = []

        for i, film_sub_ind in enumerate(film_and_substrate_inds):
            film_ind = film_sub_ind[0]
            sub_ind = film_sub_ind[1]

            interface_dir = join(
                base_dir, f"film{film_ind:02d}_sub{sub_ind:02d}"
            )

            if not isdir(interface_dir):
                os.mkdir(interface_dir)

            film = film_generator[film_ind]
            sub = substrate_generator[sub_ind]

            film_surface_charge = film.bottom_surface_charge
            sub_surface_charge = sub.top_surface_charge
            surface_charges.append([film_surface_charge, sub_surface_charge])

            film.write_file(join(interface_dir, f"POSCAR_film_{film_ind}"))
            sub.write_file(join(interface_dir, f"POSCAR_substrate_{sub_ind}"))

            interface_generator = InterfaceGenerator(
                substrate=sub,
                film=film,
                max_strain=self._max_strain,
                max_area_mismatch=self._max_area_mismatch,
                max_area=self._max_area,
                interfacial_distance=2.0,
                vacuum=60,
                center=True,
                substrate_strain_fraction=self._substrate_strain_fraction,
            )

            # Generate the interfaces
            interfaces = interface_generator.generate_interfaces()
            interface = interfaces[0]

            if i == 0:
                interface.plot_interface(
                    output=join(base_dir, "interface_view.png")
                )

            intE_matcher = IonicSurfaceMatcher(
                interface=interface,
                auto_determine_born_n=self._auto_determine_born_n,
                born_n=self._born_n,
                grid_density=self._grid_density_PES,
            )

            _ = intE_matcher.optimizePSO(
                z_bounds=self._z_bounds_PSO,
                max_iters=self._max_iterations_PSO,
                n_particles=self._n_particles_PSO,
            )
            intE_matcher.get_optimized_structure()
            opt_d_pso = interface.interfacial_distance

            intE_matcher.run_surface_matching(
                output=join(interface_dir, "PES_opt.png"),
                fontsize=14,
                cmap=self._cmap_PES,
            )
            intE_matcher.get_optimized_structure()

            intE_matcher.run_z_shift(
                interfacial_distances=np.linspace(
                    max(1.0, opt_d_pso - 2.0),
                    opt_d_pso + 2.0,
                    31,
                ),
                output=join(interface_dir, "z_shift.png"),
            )
            intE_matcher.get_optimized_structure()

            interface.write_file(join(interface_dir, "POSCAR_interface"))

            opt_d = interface.interfacial_distance
            a_shift = np.mod(interface._a_shift, 1.0)
            b_shift = np.mod(interface._b_shift, 1.0)

            opt_abz_shifts.append([a_shift, b_shift, opt_d])

            adh_energy, int_energy = intE_matcher.get_current_energy()
            film_surface_energy = intE_matcher.film_surface_energy
            sub_surface_energy = intE_matcher.sub_surface_energy
            surface_energies.append([film_surface_energy, sub_surface_energy])

            energies.append([int_energy, adh_energy])
            print("")

        energies = np.array(energies)
        opt_abz_shifts = np.array(opt_abz_shifts)
        film_sub_inds = np.array(film_and_substrate_inds)
        surface_charges = np.round(np.array(surface_charges), 4)
        surface_energies = np.array(surface_energies)

        data = {
            "Film Index": film_sub_inds[:, 0].astype(int),
            "Substrate Index": film_sub_inds[:, 1].astype(int),
            "a Shift": opt_abz_shifts[:, 0],
            "b Shift": opt_abz_shifts[:, 1],
            "Interfacial Distance": opt_abz_shifts[:, 2],
            "Film Surface Charge": surface_charges[:, 0],
            "Substrate Surface Charge": surface_charges[:, 1],
            "Score Film Surface Energy": surface_energies[:, 0],
            "Score Substrate Surface Energy": surface_energies[:, 1],
            "Score Interface Energy": energies[:, 0],
            "Score Adhesion Energy": energies[:, 1],
        }

        df = pd.DataFrame(data=data)
        df.to_csv(join(base_dir, "opt_data.csv"), index=False)

        x_label_key = "(Film Index, Substrate Index)"
        df[x_label_key] = [
            f"({int(row['Film Index'])},{int(row['Substrate Index'])})"
            for i, row in df.iterrows()
        ]

        intE_key = "Interface Energy (eV/${\\AA}^{2}$)"
        intE_df = df[[x_label_key, "Score Interface Energy"]].copy()
        intE_df.columns = [x_label_key, intE_key]
        intE_df.sort_values(by=intE_key, inplace=True)

        adhE_key = "Adhesion Energy (eV/${\\AA}^{2}$)"
        adhE_df = df[[x_label_key, "Score Adhesion Energy"]].copy()
        adhE_df.columns = [x_label_key, adhE_key]
        adhE_df.sort_values(by=adhE_key, inplace=True)

        fig, (ax_adh, ax_int) = plt.subplots(
            figsize=(max(len(df) / 3, 7), 7),
            dpi=400,
            nrows=2,
        )

        ax_adh.tick_params(axis="x", rotation=90.0)
        ax_int.tick_params(axis="x", rotation=90.0)
        ax_adh.axhline(y=0, color="black", linewidth=0.5)
        ax_int.axhline(y=0, color="black", linewidth=0.5)

        sns.barplot(
            data=adhE_df,
            x=x_label_key,
            y=adhE_key,
            color="lightgrey",
            edgecolor="black",
            linewidth=0.5,
            ax=ax_adh,
        )
        sns.barplot(
            data=intE_df,
            x=x_label_key,
            y=intE_key,
            color="lightgrey",
            edgecolor="black",
            linewidth=0.5,
            ax=ax_int,
        )

        fig.tight_layout(pad=0.5)
        fig.savefig(join(base_dir, "opt_energies.png"))

        plt.close(fig=fig)
