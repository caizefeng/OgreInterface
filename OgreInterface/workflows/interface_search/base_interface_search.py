from abc import ABC
import typing as tp
from os.path import isdir, join, abspath
import os
import json

from pymatgen.core.structure import Structure
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from OgreInterface.generate import InterfaceGenerator, BaseSurfaceGenerator
from OgreInterface.surface_matching import (
    BaseSurfaceMatcher,
    BaseSurfaceEnergy,
)
from OgreInterface.surfaces import BaseSurface
from OgreInterface import utils


class BaseInterfaceSearch(ABC):
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
        surface_matching_module: BaseSurfaceMatcher,
        surface_energy_module: BaseSurfaceEnergy,
        surface_generator: BaseSurfaceGenerator,
        substrate_bulk: tp.Union[Structure, Atoms, str],
        film_bulk: tp.Union[Structure, Atoms, str],
        substrate_miller_index: tp.List[int],
        film_miller_index: tp.List[int],
        surface_matching_kwargs: tp.Dict[str, tp.Any] = {},
        surface_energy_kwargs: tp.Dict[str, tp.Any] = {},
        minimum_slab_thickness: float = 18.0,
        max_strain: float = 0.01,
        max_area_mismatch: tp.Optional[float] = None,
        max_area: tp.Optional[float] = None,
        substrate_strain_fraction: float = 0.0,
        refine_structure: bool = True,
        suppress_warnings: bool = True,
        n_particles_PSO: int = 20,
        max_iterations_PSO: int = 150,
        z_bounds_PSO: tp.Optional[tp.List[float]] = None,
        grid_density_PES: float = 2.5,
        use_most_stable_substrate: bool = True,
        cmap_PES="coolwarm",
    ):
        self.surface_matching_module = surface_matching_module
        self.surface_energy_module = surface_energy_module
        self.surface_generator = surface_generator
        self.surface_matching_kwargs = surface_matching_kwargs
        self.surface_energy_kwargs = surface_energy_kwargs
        self._refine_structure = refine_structure
        self._suppress_warnings = suppress_warnings
        if type(substrate_bulk) is str:
            self._substrate_bulk = utils.load_bulk(
                atoms_or_structure=Structure.from_file(substrate_bulk),
                refine_structure=self._refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
        else:
            self._substrate_bulk = utils.load_bulk(
                atoms_or_structure=substrate_bulk,
                refine_structure=self._refine_structure,
                suppress_warnings=self._suppress_warnings,
            )

        if type(film_bulk) is str:
            self._film_bulk = utils.load_bulk(
                atoms_or_structure=Structure.from_file(film_bulk),
                refine_structure=self._refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
        else:
            self._film_bulk = utils.load_bulk(
                atoms_or_structure=film_bulk,
                refine_structure=self._refine_structure,
                suppress_warnings=self._suppress_warnings,
            )

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

    def _get_surface_generators(self):
        substrate_generator = self.surface_generator(
            bulk=self._substrate_bulk,
            miller_index=self._substrate_miller_index,
            layers=None,
            minimum_thickness=self._minimum_slab_thickness,
            vacuum=40.0,
            refine_structure=self._refine_structure,
        )

        film_generator = self.surface_generator(
            bulk=self._film_bulk,
            miller_index=self._film_miller_index,
            layers=None,
            minimum_thickness=self._minimum_slab_thickness,
            vacuum=40.0,
            refine_structure=True,
        )

        return substrate_generator, film_generator

    def _get_most_stable_surface(
        self, surface_generator: BaseSurfaceGenerator
    ) -> tp.List[int]:
        surface_energies = []
        for surface in surface_generator:
            surfE_calculator = self.surface_energy_module(
                surface=surface,
                **self.surface_energy_kwargs,
            )
            surface_energies.append(surfE_calculator.get_cleavage_energy())

        surface_energies = np.round(np.array(surface_energies), 6)
        min_surface_energy = surface_energies.min()

        most_stable_indices = np.where(surface_energies == min_surface_energy)

        return most_stable_indices[0]

    def _get_film_and_substrate_inds(
        self,
        film_generator: BaseSurfaceGenerator,
        substrate_generator: BaseSurfaceGenerator,
    ) -> tp.List[tp.Tuple[int, int]]:
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
                    film_and_substrate_inds.append((i, j))

        return film_and_substrate_inds

    def run_surface_generator_methods(
        self,
        film_generator: BaseSurfaceGenerator,
        substrate_generator: BaseSurfaceGenerator,
        base_dir: str,
    ) -> tp.Any:
        pass

    def run_surface_methods(
        self,
        film: BaseSurface,
        substrate: BaseSurface,
    ) -> tp.Dict[str, tp.Any]:
        return {}

    def run_interface_search(
        self,
        filter_on_charge: bool = True,
        output_folder: str = None,
    ):
        sub_comp = self._substrate_bulk.composition.reduced_formula
        film_comp = self._film_bulk.composition.reduced_formula
        sub_miller = "".join([str(i) for i in self._substrate_miller_index])
        film_miller = "".join([str(i) for i in self._film_miller_index])
        if output_folder is None:
            base_dir = f"{film_comp}{film_miller}_{sub_comp}{sub_miller}"

            current_dirs = [d for d in os.listdir() if base_dir in d]

            if len(current_dirs) > 0:
                base_dir += f"_{len(current_dirs)}"
        else:
            base_dir = output_folder

        if not isdir(base_dir):
            os.mkdir(base_dir)

        substrate_generator, film_generator = self._get_surface_generators()

        self.run_surface_generator_methods(
            film_generator=film_generator,
            substrate_generator=substrate_generator,
            base_dir=base_dir,
        )

        film_and_substrate_inds = self._get_film_and_substrate_inds(
            film_generator=film_generator,
            substrate_generator=substrate_generator,
        )

        print(
            f"Preparing to Optimize {len(film_and_substrate_inds)} {film_comp}({film_miller})/{sub_comp}({sub_miller}) Interfaces..."
        )

        data_list = []
        plotting_data_list = []

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

            surface_specific_props = self.run_surface_methods(
                substrate=sub,
                film=film,
            )

            film.write_file(join(interface_dir, f"POSCAR_film_{film_ind:02d}"))
            sub.write_file(join(interface_dir, f"POSCAR_sub_{sub_ind:02d}"))

            interface_generator = InterfaceGenerator(
                substrate=sub,
                film=film,
                max_strain=self._max_strain,
                max_area_mismatch=self._max_area_mismatch,
                max_area=self._max_area,
                interfacial_distance=2.0,
                vacuum=60.0,
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

            surface_matcher = self.surface_matching_module(
                interface=interface,
                grid_density=self._grid_density_PES,
                **self.surface_matching_kwargs,
            )

            if self._z_bounds_PSO is None:
                min_z = 0.5
                max_z = 1.1 * surface_matcher._get_max_z()
            else:
                min_z = self._z_bounds_PSO[0]
                max_z = self._z_bounds_PSO[1]

            _ = surface_matcher.optimizePSO(
                z_bounds=self._z_bounds_PSO,
                max_iters=self._max_iterations_PSO,
                n_particles=self._n_particles_PSO,
            )
            surface_matcher.get_optimized_structure()

            opt_d_pso = interface.interfacial_distance

            surface_matcher.run_surface_matching(
                output=join(interface_dir, "PES_opt.png"),
                fontsize=14,
                cmap=self._cmap_PES,
            )

            surface_matcher.get_optimized_structure()

            surface_matcher.run_z_shift(
                interfacial_distances=np.linspace(
                    max(min_z, opt_d_pso - 2.0),
                    min(opt_d_pso + 2.0, max_z),
                    31,
                ),
                output=join(interface_dir, "z_shift.png"),
            )

            surface_matcher.get_optimized_structure()

            interface.write_file(
                join(
                    interface_dir,
                    f"POSCAR_interface_film{film_ind:02d}_sub{sub_ind:02d}",
                )
            )

            opt_d = interface.interfacial_distance
            a_shift = np.mod(interface._a_shift, 1.0)
            b_shift = np.mod(interface._b_shift, 1.0)

            adh_energy, int_energy = surface_matcher.get_current_energy()
            film_surface_energy = surface_matcher.film_surface_energy
            sub_surface_energy = surface_matcher.sub_surface_energy

            interface_structure = interface.get_interface(orthogonal=True)

            plotting_data = {
                "filmIndex": int(film_ind),
                "substrateIndex": int(sub_ind),
                "interfaceEnergy": float(int_energy),
                "adhesionEnergy": float(adh_energy),
                "aShift": float(a_shift),
                "bShift": float(b_shift),
                "interfacialDistance": float(opt_d),
                "filmSurfaceEnergy": float(film_surface_energy),
                "substrateSurfaceEnergy": float(sub_surface_energy),
            }
            plotting_data_list.append(plotting_data)

            iter_data = {
                "pesFigurePath": abspath(join(interface_dir, "PES_opt.png")),
                "zShiftFigurePath": abspath(
                    join(interface_dir, "z_shift.png")
                ),
                "interfaceStructure": interface_structure.as_dict(),
            }
            iter_data.update(surface_specific_props)
            iter_data.update(plotting_data)

            data_list.append(iter_data)
            print("")

        run_data = {
            "substrateBulk": self._substrate_bulk.as_dict(),
            "filmBulk": self._film_bulk.as_dict(),
            "filmMillerIndex": list(self._film_miller_index),
            "substrateMillerIndex": list(self._substrate_miller_index),
            "maxStrain": float(self._max_strain),
            "maxArea": self._max_area and float(self._max_area),
            "maxAreaMismatch": self._max_area_mismatch
            and float(self._max_area_mismatch),
            "optData": data_list,
        }

        with open(join(base_dir, "run_data.json"), "w") as f:
            json_str = json.dumps(run_data, indent=4, sort_keys=True)
            f.write(json_str)

        df = pd.DataFrame(data=plotting_data_list)
        df.to_csv(join(base_dir, "opt_data.csv"), index=False)

        x_label_key = "(Film Index, Substrate Index)"
        df[x_label_key] = [
            f"({int(row['filmIndex'])},{int(row['substrateIndex'])})"
            for i, row in df.iterrows()
        ]

        intE_key = "Interface Energy (eV/${\\AA}^{2}$)"
        intE_df = df[[x_label_key, "interfaceEnergy"]].copy()
        intE_df.columns = [x_label_key, intE_key]
        intE_df.sort_values(by=intE_key, inplace=True)

        adhE_key = "Adhesion Energy (eV/${\\AA}^{2}$)"
        adhE_df = df[[x_label_key, "adhesionEnergy"]].copy()
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
        fig.savefig(join(base_dir, "opt_energies.png"), transparent=False)

        plt.close(fig=fig)
