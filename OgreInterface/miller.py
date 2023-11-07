from itertools import product
from typing import Union, Optional, List

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from OgreInterface.generate import SurfaceGenerator
from OgreInterface.surfaces import OrientedBulk
from OgreInterface.lattice_match import ZurMcGill
from OgreInterface import utils


class MillerSearch(object):
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
        max_area_mismatch: Area ratio mismatch tolerance for the InterfaceGenerator
        max_angle_strain: Angle strain tolerance for the InterfaceGenerator
        max_linear_strain: Lattice vectors length mismatch tolerance for the InterfaceGenerator
        max_area: Maximum area of the matched supercells
        refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
            This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
            users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
            work exclusively with the primitive structure so we always have it on hand.

    Attributes:
        substrate (Structure): Pymatgen Structure of the substrate
        film (Structure): Pymatgen Structure of the film
        max_substrate_index (int): Max miller index of the substrate surfaces
        max_film_index (int): Max miller index of the film surfaces
        max_area_mismatch (float): Area ratio mismatch tolerance for the InterfaceGenerator
        max_angle_strain (float): Angle strain tolerance for the InterfaceGenerator
        max_linear_strain (float): Lattice vectors length mismatch tolerance for the InterfaceGenerator
        max_area (float): Maximum area of the matched supercells
        refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
            This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
            users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
            work exclusively with the primitive structure so we always have it on hand.
        substrate_inds (list): List of unique substrate surface miller indices
        film_inds (list): List of unique film surface miller indices
    """

    def __init__(
        self,
        substrate: Union[Structure, Atoms, str],
        film: Union[Structure, Atoms, str],
        max_substrate_index: int = 1,
        max_film_index: int = 1,
        max_strain: float = 0.01,
        max_area_mismatch: Optional[float] = None,
        max_area: Optional[float] = None,
        max_area_scale_factor: float = 4.1,
        refine_structure: bool = True,
        suppress_warnings: bool = False,
        custom_film_miller_indices: Optional[List[List[int]]] = None,
        custom_substrate_miller_indices: Optional[List[List[int]]] = None,
    ) -> None:
        self.refine_structure = refine_structure
        self._suppress_warnings = suppress_warnings

        if type(substrate) is str:
            self.substrate = utils.load_bulk(
                atoms_or_structure=Structure.from_file(substrate),
                refine_structure=self.refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
            # self.substrate, _ = self._get_bulk(Structure.from_file(substrate))
        else:
            self.substrate = utils.load_bulk(
                atoms_or_structure=substrate,
                refine_structure=self.refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
            # self.substrate, _ = self._get_bulk(substrate)

        if type(film) is str:
            self.film = utils.load_bulk(
                atoms_or_structure=Structure.from_file(film),
                refine_structure=self.refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
            # self.film, _ = self._get_bulk(Structure.from_file(film))
        else:
            self.film = utils.load_bulk(
                atoms_or_structure=film,
                refine_structure=self.refine_structure,
                suppress_warnings=self._suppress_warnings,
            )
            # self.film, _ = self._get_bulk(film)

        self.max_film_index = max_film_index
        self.max_substrate_index = max_substrate_index
        self.max_area_mismatch = max_area_mismatch
        self.max_strain = max_strain
        self.max_area = max_area
        self.max_area_scale_factor = max_area_scale_factor

        if custom_substrate_miller_indices is not None:
            self.substrate_inds = custom_substrate_miller_indices
        else:
            self.substrate_inds = utils.get_unique_miller_indices(
                self.substrate,
                self.max_substrate_index,
            )

        if custom_film_miller_indices is not None:
            self.film_inds = custom_film_miller_indices
        else:
            self.film_inds = utils.get_unique_miller_indices(
                self.film,
                self.max_film_index,
            )

        self._misfit_data = None
        self._area_data = None

    def run_scan(self) -> None:
        """
        Run the miller index scan by looping through all combinations of unique surface miller indices
        for the substrate and film.
        """
        substrates = []
        films = []

        for inds in self.substrate_inds:
            sub_obs = OrientedBulk(
                bulk=self.substrate,
                miller_index=inds,
                make_planar=True,
            )
            sub_inplane_vectors = sub_obs.inplane_vectors
            sub_area = sub_obs.area
            sub_basis = sub_obs.crystallographic_basis

            substrates.append([sub_inplane_vectors, sub_area, sub_basis])

        for inds in self.film_inds:
            film_obs = OrientedBulk(
                bulk=self.film,
                miller_index=inds,
                make_planar=True,
            )
            film_inplane_vectors = film_obs.inplane_vectors
            film_area = film_obs.area
            film_basis = film_obs.crystallographic_basis

            films.append([film_inplane_vectors, film_area, film_basis])

        misfits = np.ones((len(substrates), len(films))) * np.nan
        areas = np.ones((len(substrates), len(films))) * np.nan

        for i, substrate in enumerate(substrates):
            for j, film in enumerate(films):
                zm = ZurMcGill(
                    film_vectors=film[0],
                    substrate_vectors=substrate[0],
                    film_basis=film[2],
                    substrate_basis=substrate[2],
                    max_area=self.max_area,
                    max_strain=self.max_strain,
                    max_area_mismatch=self.max_area_mismatch,
                    max_area_scale_factor=self.max_area_scale_factor,
                )
                matches = zm.run()

                if len(matches) > 0:
                    min_area_match = matches[0]
                    area = min_area_match.area
                    strain = min_area_match.strain
                    misfits[i, j] = strain
                    areas[i, j] = area / np.sqrt(substrate[1] * film[1])

        self.misfits = np.round(misfits.T, 8)
        self.areas = areas.T

    def plot_misfits(
        self,
        cmap: str = "coolwarm",
        dpi: int = 400,
        output: str = "misfit_plot.png",
        fontsize: float = 12.0,
        figure_scale: float = 1.0,
        labelrotation: float = -20.0,
        substrate_label: Union[str, None] = None,
        film_label: Union[str, None] = None,
        show_in_colab: bool = False,
    ) -> None:
        """
        Plot the results of the miller index scan.

        Args:
            cmap: color map (matplotlib)
            dpi: dpi (dots per inch) of the output image.
                Setting dpi=100 gives reasonably sized images when viewed in colab notebook
            output: File path for the output image
            fontsize: fontsize for axis and tick labels
            figure_scale: The figure size is automatically changed to fit the ratio of the substrate / film indices
                but in some cases, especially with large amounts of unique surfaces the figure size needs to be increased.
                This should usually stay at 1.0.
            labelrotation: Determines how much the labels on the x-axis should be rotated. This is usefull to avoid overlapping labels
            substrate_label: If none, this is automatically determined using the reduced formula of the bulk structure
            film_label: If none, this is automatically determined using the reduced formula of the bulk structure
            show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
                if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
        """
        ylabels = []
        for ylabel in self.film_inds:
            tmp_label = utils.get_miller_index_label(ylabel)
            ylabels.append(f"({tmp_label})")

        xlabels = []
        for xlabel in self.substrate_inds:
            tmp_label = utils.get_miller_index_label(xlabel)
            xlabels.append(f"({tmp_label})")

        N = len(self.film_inds)
        M = len(self.substrate_inds)
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        s = self.areas
        c = self.misfits * 100

        if (M / N) < 1.0:
            figsize = (figure_scale * 5, (N / M) * figure_scale * 4)
        else:
            figsize = (figure_scale * 5 * (M / N), figure_scale * 4)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_divider = make_axes_locatable(ax)

        cax = ax_divider.append_axes(
            "right",
            size=np.min(figsize) * 0.04,
            pad=np.min(figsize) * 0.01,
        )

        if film_label is None:
            film_label = utils.get_latex_formula(
                self.film.composition.reduced_formula
            )

        ax.set_ylabel(film_label + " Miller Index", fontsize=fontsize)

        if substrate_label is None:
            substrate_label = utils.get_latex_formula(
                self.substrate.composition.reduced_formula
            )

        ax.set_xlabel(substrate_label + " Miller Index", fontsize=fontsize)

        if not np.isnan(s).all():
            R = 0.85 * s / np.nanmax(s) / 2
        else:
            print(
                "WARNING: No matches were found with the current settings. Try increasing max_area or max_strain"
            )
            R = s

        circles = [
            plt.Circle((i, j), radius=r, edgecolor="black", lw=3)
            for r, i, j in zip(R.flat, x.flat, y.flat)
        ]
        col = PatchCollection(
            circles,
            array=c.flatten(),
            cmap=cmap,
            norm=Normalize(
                vmin=0.0,
                vmax=100 * self.max_strain,
            ),
            edgecolor="black",
            linewidth=1,
        )
        ax.add_collection(col)

        ax.set(
            xticks=np.arange(M),
            yticks=np.arange(N),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
        ax.tick_params(axis="x", labelrotation=labelrotation)
        ax.tick_params(labelsize=fontsize)
        ax.grid(which="minor", linestyle=":", linewidth=0.75)

        cbar = fig.colorbar(col, cax=cax)
        cbar.set_label("Strain (%)", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.ticklabel_format(
            style="sci", scilimits=(-3, 3), useMathText=True
        )
        cbar.ax.yaxis.set_offset_position("left")

        ax.set_aspect("equal")
        fig.tight_layout(pad=0.4)
        fig.savefig(output, bbox_inches="tight", transparent=False)

        if not show_in_colab:
            plt.close(fig)


if __name__ == "__main__":
    ms = MillerSearch(
        substrate="./dd-poscars/POSCAR_InAs_conv",
        film="./dd-poscars/POSCAR_Al_conv",
        max_film_index=2,
        max_substrate_index=2,
        max_linear_strain=0.01,
        max_angle_strain=0.01,
        max_area_mismatch=0.01,
        max_area=500,
    )
    ms.run_scan()
    ms.plot_misfits(figsize=(6.5, 5), fontsize=17, labelrotation=0)
