from abc import ABC
import typing as tp
from os.path import isdir, join, abspath
import os
import io
import base64
import json
from multiprocessing import Pool
import logging
import itertools

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.composition import Composition
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

from OgreInterface.interfaces import BaseInterface
from OgreInterface.generate import InterfaceGenerator, BaseSurfaceGenerator
from OgreInterface.surface_matching import (
    BaseSurfaceMatcher,
    BaseSurfaceEnergy,
)
from OgreInterface.surfaces import BaseSurface
from OgreInterface import utils

matplotlib.use("agg")


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
        vacuum: float = 60.0,
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
        cmap_PES: str = "coolwarm",
        n_workers: int = 1,
        app_mode: bool = False,
        dpi: int = 400,
        verbose: bool = True,
        fast_mode: bool = True,
    ):
        if n_workers > 1:
            self._verbose = False
        else:
            self._verbose = verbose

        self._fast_mode = fast_mode
        self.surface_matching_module = surface_matching_module
        self.surface_energy_module = surface_energy_module
        self.surface_generator = surface_generator
        self.surface_matching_kwargs = surface_matching_kwargs
        self.surface_energy_kwargs = surface_energy_kwargs
        self._refine_structure = refine_structure
        self._suppress_warnings = suppress_warnings
        self.n_workers = n_workers
        self._app_mode = app_mode
        self._dpi = dpi
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
        self._vacuum = vacuum
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

    def _calc_surface_energy(self, surface):
        surfE_calculator = self.surface_energy_module(
            surface=surface,
            **self.surface_energy_kwargs,
        )

        return surfE_calculator.get_cleavage_energy()

    def _get_most_stable_surface(
        self, surface_generator: BaseSurfaceGenerator
    ) -> tp.List[int]:
        if self.n_workers <= 1:
            surface_energies = [
                self._calc_surface_energy(surface)
                for surface in surface_generator
            ]
        else:
            with Pool(self.n_workers) as p:
                surface_energies = p.map(
                    self._calc_surface_energy,
                    surface_generator,
                )

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

    def _optimize_single_interface(
        self,
        base_dir: str,
        interface: BaseInterface,
    ):
        film = interface.film
        sub = interface.substrate

        film_ind = film.termination_index
        sub_ind = sub.termination_index

        data = {
            "filmIndex": int(film_ind),
            "substrateIndex": int(sub_ind),
        }

        interface_dir = join(base_dir, f"film{film_ind:02d}_sub{sub_ind:02d}")

        if not self._app_mode:
            if not isdir(interface_dir):
                os.mkdir(interface_dir)

        surface_specific_props = self.run_surface_methods(
            substrate=sub,
            film=film,
        )

        data.update(surface_specific_props)

        if not self._app_mode:
            film.write_file(join(interface_dir, f"POSCAR_film_{film_ind:02d}"))
            sub.write_file(join(interface_dir, f"POSCAR_sub_{sub_ind:02d}"))

        surface_matcher = self.surface_matching_module(
            interface=interface,
            grid_density=self._grid_density_PES,
            verbose=self._verbose,
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

        if not self._fast_mode:
            stream_PES = io.BytesIO()
            surface_matcher.run_surface_matching(
                output=stream_PES,
                # output=join(interface_dir, "PES_opt.png"),
                fontsize=14,
                cmap=self._cmap_PES,
                dpi=self._dpi,
            )
            stream_PES_value = stream_PES.getvalue()
            stream_PES_base64 = base64.b64encode(stream_PES_value).decode()

            data["pesFigure"] = stream_PES_base64

            if not self._app_mode:
                with open(join(interface_dir, "PES_opt.png"), "wb") as f:
                    f.write(stream_PES_value)

        surface_matcher.get_optimized_structure()

        if not self._fast_mode:
            stream_z_shift = io.BytesIO()
            surface_matcher.run_z_shift(
                interfacial_distances=np.linspace(
                    max(min_z, opt_d_pso - 2.0),
                    min(opt_d_pso + 2.0, max_z),
                    31,
                ),
                output=stream_z_shift,
                dpi=self._dpi,
                zoom_to_minimum=True,
            )

            stream_z_shift_value = stream_z_shift.getvalue()
            stream_z_shift_base64 = base64.b64encode(
                stream_z_shift_value
            ).decode()

            data["zShiftFigure"] = stream_z_shift_base64

            if not self._app_mode:
                with open(join(interface_dir, "z_shift.png"), "wb") as f:
                    f.write(stream_z_shift_value)

        surface_matcher.get_optimized_structure()

        opt_d = interface.interfacial_distance
        a_shift = np.mod(interface._a_shift, 1.0)
        b_shift = np.mod(interface._b_shift, 1.0)

        adh_energy, int_energy = surface_matcher.get_current_energy()
        film_surface_energy = surface_matcher.film_surface_energy
        sub_surface_energy = surface_matcher.sub_surface_energy

        interface_structure = interface.get_interface(orthogonal=True)

        small_interface_structure = self._get_layers_around_interface(
            interface,
        )

        (
            film_termination,
            substrate_termination,
        ) = self._get_terminations_from_small_interface(
            structure=small_interface_structure,
        )

        if not self._app_mode:
            interface.write_file(
                join(
                    interface_dir,
                    f"POSCAR_interface_film{film_ind:02d}_sub{sub_ind:02d}",
                )
            )
            Poscar(small_interface_structure).write_file(
                join(
                    interface_dir,
                    f"POSCAR_interface_film{film_ind:02d}_sub{sub_ind:02d}_small",
                )
            )

        interface_structure.lattice._pbc = (True, True, False)
        small_interface_structure.lattice._pbc = (True, True, False)

        data["interfaceEnergy"] = float(int_energy)
        data["adhesionEnergy"] = float(adh_energy)
        data["aShift"] = float(a_shift)
        data["bShift"] = float(b_shift)
        data["interfacialDistance"] = float(opt_d)
        data["filmSurfaceEnergy"] = float(film_surface_energy)
        data["substrateSurfaceEnergy"] = float(sub_surface_energy)
        data["fullInterfaceStructure"] = interface_structure.as_dict()
        data["smallInterfaceStructure"] = small_interface_structure.as_dict()
        data["filmTerminationComp"] = film_termination
        data["substrateTerminationComp"] = substrate_termination
        data["area"] = interface.area
        data["strain"] = 100 * interface.match.strain

        return data

    def _get_terminations_from_small_interface(
        self, structure: Structure
    ) -> tp.Tuple[str, str]:
        is_film = np.array(structure.site_properties["is_film"])
        is_sub = np.logical_not(is_film)
        atomic_numbers = np.array(structure.atomic_numbers).astype(int)
        unique_film_numbers, unique_film_counts = np.unique(
            atomic_numbers[is_film], return_counts=True
        )
        unique_sub_numbers, unique_sub_counts = np.unique(
            atomic_numbers[is_sub], return_counts=True
        )

        film_comp = Composition(
            {n: c for n, c in zip(unique_film_numbers, unique_film_counts)}
        )

        sub_comp = Composition(
            {n: c for n, c in zip(unique_sub_numbers, unique_sub_counts)}
        )

        return film_comp.reduced_formula, sub_comp.reduced_formula

    def _get_layers_around_interface(
        self,
        interface: BaseInterface,
        max_thickness: float = 4.0,
    ) -> Structure:
        structure = interface.get_interface(orthogonal=True)

        is_film = np.array(structure.site_properties["is_film"])
        is_sub = np.logical_not(is_film)
        atomic_layers = np.array(
            structure.site_properties["atomic_layer_index"]
        )

        z_coords = structure.cart_coords[:, -1]

        film_bottom = z_coords[is_film].min()
        sub_top = z_coords[is_sub].max()

        rel_sub_coords = sub_top - z_coords
        rel_film_coords = z_coords - film_bottom

        sub_mask = (rel_sub_coords <= max_thickness) & is_sub
        film_mask = (rel_film_coords <= max_thickness) & is_film

        n_sub_layers = (
            atomic_layers[is_sub].max() - atomic_layers[sub_mask].min()
        )
        n_film_layers = atomic_layers[film_mask].max()

        inds_to_keep = []

        for i in range(n_sub_layers):
            sub_inds = interface.get_substrate_layer_indices(
                layer_from_interface=i,
                atomic_layers=True,
            )
            inds_to_keep.append(sub_inds)

        for i in range(n_film_layers):
            film_inds = interface.get_film_layer_indices(
                layer_from_interface=i,
                atomic_layers=True,
            )

            inds_to_keep.append(film_inds)

        inds_to_keep = np.concatenate(inds_to_keep)
        all_inds = np.ones(len(structure)).astype(bool)
        all_inds[inds_to_keep] = False

        inds_to_delete = np.where(all_inds)[0]

        small_structure = structure.copy()
        small_structure.remove_sites(inds_to_delete)

        return small_structure

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

        if not self._app_mode:
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

        if self._verbose:
            print(
                f"Preparing to Optimize {len(film_and_substrate_inds)} {film_comp}({film_miller})/{sub_comp}({sub_miller}) Interfaces..."
            )

        interfaces = []

        for i, film_sub_ind in enumerate(film_and_substrate_inds):
            film_ind = film_sub_ind[0]
            sub_ind = film_sub_ind[1]

            film = film_generator[film_ind]
            sub = substrate_generator[sub_ind]

            interface_generator = InterfaceGenerator(
                substrate=sub,
                film=film,
                max_strain=self._max_strain,
                max_area_mismatch=self._max_area_mismatch,
                max_area=self._max_area,
                interfacial_distance=2.0,
                vacuum=self._vacuum,
                center=True,
                substrate_strain_fraction=self._substrate_strain_fraction,
                verbose=self._verbose,
            )

            # Generate the interfaces
            ifaces = interface_generator.generate_interfaces(
                generate_all=False
            )
            iface = ifaces[0]

            interfaces.append(iface)

            if i == 0:
                stream_view = io.BytesIO()
                iface.plot_interface(
                    output=stream_view,
                )

                stream_view_value = stream_view.getvalue()
                stream_view_base64 = base64.b64encode(
                    stream_view_value
                ).decode()

                if not self._app_mode:
                    with open(join(base_dir, "interface_view.png"), "wb") as f:
                        f.write(stream_view_value)

        if self.n_workers <= 1:
            data_list = []
            for interface in interfaces:
                data = self._optimize_single_interface(
                    base_dir=base_dir,
                    interface=interface,
                )
                data_list.append(data)
        else:
            with Pool(self.n_workers) as p:
                inputs = zip(itertools.repeat(base_dir), interfaces)
                data_list = p.starmap(self._optimize_single_interface, inputs)

        data_list.sort(key=lambda x: x["interfaceEnergy"])

        df = pd.DataFrame(data=data_list)
        df = df[
            [
                "filmIndex",
                "substrateIndex",
                "interfaceEnergy",
                "adhesionEnergy",
                "aShift",
                "bShift",
                "interfacialDistance",
                "filmSurfaceEnergy",
                "substrateSurfaceEnergy",
            ]
        ]

        if not self._app_mode:
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
            dpi=self._dpi,
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

        stream_energies = io.BytesIO()
        fig.savefig(
            stream_energies,
            transparent=False,
        )

        plt.close(fig=fig)

        stream_energies_value = stream_energies.getvalue()
        stream_energies_base64 = base64.b64encode(
            stream_energies_value
        ).decode()

        if not self._app_mode:
            with open(join(base_dir, "opt_energies.png"), "wb") as f:
                f.write(stream_energies_value)

        film_miller = list(self._film_miller_index)
        substrate_miller = list(self._substrate_miller_index)

        run_data = {
            "bulkData": {
                "substrateBulk": self._substrate_bulk.to_json(),
                "filmBulk": self._film_bulk.to_json(),
            },
            "optData": {
                "filmMillerIndex": film_miller,
                "substrateMillerIndex": substrate_miller,
                "maxStrain": float(100 * self._max_strain),
                "maxArea": self._max_area and float(self._max_area),
                "maxAreaMismatch": self._max_area_mismatch
                and float(self._max_area_mismatch),
                "matchFigure": stream_view_base64,
                "totalEnergiesFigure": stream_energies_base64,
                "interfaceData": data_list,
            },
        }

        # TODO: return data if in app mode
        if self._app_mode:
            return run_data["optData"]
        else:
            with open(join(base_dir, "run_data.json"), "w") as f:
                json_str = json.dumps(run_data, indent=4, sort_keys=True)
                f.write(json_str)
