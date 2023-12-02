import typing as tp
from os.path import join
from multiprocessing import Pool, cpu_count

from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from ase import Atoms
import numpy as np

from OgreInterface.generate import SurfaceGenerator
from OgreInterface.surfaces import Surface
from OgreInterface.interfaces import Interface
from OgreInterface.surface_matching import (
    IonicSurfaceMatcher,
    IonicSurfaceEnergy,
)
from OgreInterface.workflows.interface_search import BaseInterfaceSearch
from OgreInterface.plotting_tools import plot_surface_charge_matrix


class IonicInterfaceSearch(BaseInterfaceSearch):
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
        substrate_bulk: tp.Union[Structure, Atoms, str],
        film_bulk: tp.Union[Structure, Atoms, str],
        substrate_miller_index: tp.List[int],
        film_miller_index: tp.List[int],
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
        cmap_PES="coolwarm",
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
        n_workers: int = cpu_count(),
        app_mode: bool = False,
        dpi: int = 400,
        verbose: bool = True,
        fast_mode: bool = False,
    ):
        surface_matching_kwargs = {
            "auto_determine_born_n": auto_determine_born_n,
            "born_n": born_n,
        }
        super().__init__(
            surface_matching_module=IonicSurfaceMatcher,
            surface_energy_module=IonicSurfaceEnergy,
            surface_generator=SurfaceGenerator,
            substrate_bulk=substrate_bulk,
            film_bulk=film_bulk,
            substrate_miller_index=substrate_miller_index,
            film_miller_index=film_miller_index,
            surface_matching_kwargs=surface_matching_kwargs,
            surface_energy_kwargs=surface_matching_kwargs,
            minimum_slab_thickness=minimum_slab_thickness,
            vacuum=vacuum,
            max_strain=max_strain,
            max_area_mismatch=max_area_mismatch,
            max_area=max_area,
            substrate_strain_fraction=substrate_strain_fraction,
            refine_structure=refine_structure,
            suppress_warnings=suppress_warnings,
            n_particles_PSO=n_particles_PSO,
            max_iterations_PSO=max_iterations_PSO,
            z_bounds_PSO=z_bounds_PSO,
            grid_density_PES=grid_density_PES,
            use_most_stable_substrate=use_most_stable_substrate,
            cmap_PES=cmap_PES,
            n_workers=n_workers,
            app_mode=app_mode,
            dpi=dpi,
            verbose=verbose,
            fast_mode=fast_mode,
        )

    def _get_film_and_substrate_inds(
        self,
        film_generator: SurfaceGenerator,
        substrate_generator: SurfaceGenerator,
        filter_on_charge: bool = True,
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

    def _get_surface_atoms(
        self,
        oriented_bulk_structure: Structure,
    ) -> tp.Dict[str, np.array]:
        oxi_struc = oriented_bulk_structure.copy()
        oxi_struc.add_oxidation_state_by_site(
            oriented_bulk_structure.site_properties["charge"]
        )

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)

        surface_atom_dict = {"top": [], "bottom": []}

        # Loop through all sites in the structure to get the bonding environments
        for i, site in enumerate(oxi_struc.sites):
            # Get nearest neighbor info dict
            info_dict = cnn.get_nn_info(oxi_struc, i)

            obs_equiv = site.properties["oriented_bulk_equivalent"]

            # Loop through all the neighboring sites
            for neighbor in info_dict:
                if neighbor["image"][-1] < 0:
                    surface_atom_dict["bottom"].append(obs_equiv)

                if neighbor["image"][-1] > 0:
                    surface_atom_dict["top"].append(obs_equiv)

        for k, v in surface_atom_dict.items():
            surface_atom_dict[k] = np.unique(v)

        return surface_atom_dict

    def _get_layers_around_interface(self, interface: Interface) -> Structure:
        structure = interface.get_interface(orthogonal=True)
        sub_layers = interface.substrate.layers - 1
        site_props = structure.site_properties

        is_film = np.array(site_props["is_film"])
        is_sub = np.logical_not(is_film)
        obs_equiv = np.array(site_props["oriented_bulk_equivalent"])
        layer_index = np.array(site_props["layer_index"])

        sub_obs = interface.substrate_oriented_bulk_structure
        film_obs = interface.film_oriented_bulk_structure

        sub_surface_atoms = self._get_surface_atoms(
            oriented_bulk_structure=sub_obs
        )

        film_surface_atoms = self._get_surface_atoms(
            oriented_bulk_structure=film_obs
        )

        sub_top = (
            (layer_index == sub_layers)
            & is_sub
            & np.isin(obs_equiv, sub_surface_atoms["top"])
        )

        film_bottom = (
            (layer_index == 0)
            & is_film
            & np.isin(obs_equiv, film_surface_atoms["bottom"])
        )

        inds_to_keep = np.logical_or(sub_top, film_bottom)
        inds_to_delete = np.where(np.logical_not(inds_to_keep))[0]

        small_structure = structure.copy()
        small_structure.remove_sites(inds_to_delete)

        return small_structure

    def run_surface_generator_methods(
        self,
        film_generator: SurfaceGenerator,
        substrate_generator: SurfaceGenerator,
        base_dir: str,
    ) -> tp.Any:
        if not self._app_mode:
            plot_surface_charge_matrix(
                films=film_generator,
                substrates=substrate_generator,
                output=join(base_dir, "surface_charge_matrix.png"),
            )

    def run_surface_methods(
        self,
        film: Surface,
        substrate: Surface,
    ) -> tp.Dict[str, tp.Any]:
        return {
            "filmSurfaceCharge": float(film.bottom_surface_charge),
            "substrateSurfaceCharge": float(substrate.top_surface_charge),
        }
