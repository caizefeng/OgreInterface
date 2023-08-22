from OgreInterface.generate import SurfaceGenerator
from OgreInterface.lattice_match import ZurMcGill
from OgreInterface import utils

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices

from ase import Atoms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from functools import reduce
from itertools import product

from typing import Union


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
        max_area_mismatch: float = 0.01,
        max_angle_strain: float = 0.01,
        max_linear_strain: float = 0.01,
        max_area: float = 500.0,
        refine_structure: bool = True,
    ) -> None:
        self.refine_structure = refine_structure
        if type(substrate) == str:
            self.substrate, _ = self._get_bulk(Structure.from_file(substrate))
        else:
            self.substrate, _ = self._get_bulk(substrate)

        if type(film) == str:
            self.film, _ = self._get_bulk(Structure.from_file(film))
        else:
            self.film, _ = self._get_bulk(film)

        self.max_film_index = max_film_index
        self.max_substrate_index = max_substrate_index
        self.max_area_mismatch = max_area_mismatch
        self.max_angle_strain = max_angle_strain
        self.max_linear_strain = max_linear_strain
        self.max_area = max_area
        self.substrate_inds = self._get_unique_miller_indices(
            self.substrate, self.max_substrate_index
        )

        self.film_inds = self._get_unique_miller_indices(
            self.film, self.max_film_index
        )

        self._misfit_data = None
        self._area_data = None

    def _get_bulk(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
        else:
            raise TypeError(
                f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        if self.refine_structure:
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

    # def _get_bulk(self, atoms_or_struc):
    #     if type(atoms_or_struc) == Atoms:
    #         init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
    #     elif type(atoms_or_struc) == Structure:
    #         init_structure = atoms_or_struc
    #     else:
    #         raise TypeError(
    #             f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
    #         )

    #     if self.convert_to_conventional:
    #         sg = SpacegroupAnalyzer(init_structure)
    #         conventional_structure = sg.get_conventional_standard_structure()
    #         prim_structure = sg.get_primitive_standard_structure()

    #         return (conventional_structure, prim_structure)
    #     else:
    #         prim_structure = init_structure.get_primitive_structure()

    #         return init_structure, prim_structure

    # def _get_bulk(self, atoms_or_struc):
    #     if type(atoms_or_struc) == Atoms:
    #         init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
    #     elif type(atoms_or_struc) == Structure:
    #         init_structure = atoms_or_struc
    #     else:
    #         raise TypeError(
    #             f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
    #         )

    #     if self.convert_to_conventional:
    #         sg = SpacegroupAnalyzer(init_structure)
    #         conventional_structure = sg.get_conventional_standard_structure()
    #         conventional_atoms = AseAtomsAdaptor.get_atoms(
    #             conventional_structure
    #         )

    #         return conventional_structure, conventional_atoms
    #     else:
    #         init_atoms = AseAtomsAdaptor().get_atoms(init_structure)

    #         return init_structure, init_atoms

    def _float_gcd(self, a, b, rtol=1e-05, atol=1e-08):
        t = min(abs(a), abs(b))
        while abs(b) > rtol * t + atol:
            a, b = b, a % b
        return a

    def _hex_to_cubic(self, uvtw):
        u = 2 * uvtw[0] + uvtw[1]
        v = 2 * uvtw[1] + uvtw[0]
        w = uvtw[-1]

        output = np.array([u, v, w])
        output = utils._get_reduced_vector(output)

        return output.astype(int)

    def _cubic_to_hex(self, uvw):
        u = (1 / 3) * ((2 * uvw[0]) - uvw[1])
        v = (1 / 3) * ((2 * uvw[1]) - uvw[0])
        t = -(u + v)
        w = uvw[-1]

        output = np.array([u, v, t, w])
        output = utils._get_reduced_vector(output)

        return output.astype(int)

    def _get_unique_miller_indices(self, struc: Structure, max_index: int):
        struc_sg = SpacegroupAnalyzer(struc)
        lattice = struc.lattice
        recip = struc.lattice.reciprocal_lattice_crystallographic
        symmops = struc_sg.get_point_group_operations(cartesian=False)
        planes = set(list(product(range(-max_index, max_index + 1), repeat=3)))
        planes.remove((0, 0, 0))

        reduced_planes = []
        for plane in planes:
            reduced_plane = utils._get_reduced_vector(
                np.array(plane).astype(float)
            )
            reduced_plane = reduced_plane.astype(int)
            reduced_planes.append(tuple(reduced_plane))

        reduced_planes = set(reduced_planes)

        planes_dict = {p: [] for p in reduced_planes}

        for plane in reduced_planes:
            frac_vec = np.array(plane).dot(recip.metric_tensor)
            if plane in planes_dict.keys():
                for i, symmop in enumerate(symmops):
                    frac_point_out = symmop.apply_rotation_only(frac_vec)
                    point_out = frac_point_out.dot(lattice.metric_tensor)
                    point_out = utils._get_reduced_vector(np.round(point_out))
                    point_out = tuple(point_out.astype(int))
                    planes_dict[plane].append(point_out)
                    if point_out != plane:
                        if point_out in planes_dict.keys():
                            del planes_dict[point_out]

        unique_planes = []

        for k in planes_dict:
            equivalent_planes = np.array(list(set(planes_dict[k])))
            diff = np.abs(np.sum(np.sign(equivalent_planes), axis=1))
            like_signs = equivalent_planes[diff == np.max(diff)]
            if len(like_signs) == 1:
                unique_planes.append(like_signs[0])
            else:
                first_max = like_signs[
                    np.abs(like_signs)[:, 0]
                    == np.max(np.abs(like_signs)[:, 0])
                ]
                if len(first_max) == 1:
                    unique_planes.append(first_max[0])
                else:
                    second_max = first_max[
                        np.abs(first_max)[:, 1]
                        == np.max(np.abs(first_max)[:, 1])
                    ]
                    if len(second_max) == 1:
                        unique_planes.append(second_max[0])
                    else:
                        unique_planes.append(
                            second_max[
                                np.argmax(np.sign(second_max).sum(axis=1))
                            ]
                        )

        unique_planes = np.vstack(unique_planes)
        sorted_planes = sorted(
            unique_planes, key=lambda x: (np.linalg.norm(x), -np.sign(x).sum())
        )

        return np.vstack(sorted_planes)

    def run_scan(self) -> None:
        """
        Run the miller index scan by looping through all combinations of unique surface miller indices
        for the substrate and film.
        """
        substrates = []
        films = []

        for inds in self.substrate_inds:
            sg_sub = SurfaceGenerator(
                bulk=self.substrate,
                miller_index=inds,
                layers=5,
                vacuum=10,
                generate_all=False,
                lazy=True,
                refine_structure=self.refine_structure,
            )
            sub_inplane_vectors = sg_sub.inplane_vectors
            sub_area = np.linalg.norm(
                np.cross(sub_inplane_vectors[0], sub_inplane_vectors[1])
            )
            substrates.append(
                [sub_inplane_vectors, sub_area, sg_sub.uvw_basis]
            )

        for inds in self.film_inds:
            sg_film = SurfaceGenerator(
                bulk=self.film,
                miller_index=inds,
                layers=5,
                vacuum=10,
                generate_all=False,
                lazy=True,
                refine_structure=self.refine_structure,
            )
            film_inplane_vectors = sg_film.inplane_vectors
            film_area = np.linalg.norm(
                np.cross(film_inplane_vectors[0], film_inplane_vectors[1])
            )

            films.append([film_inplane_vectors, film_area, sg_film.uvw_basis])

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
                    max_linear_strain=self.max_linear_strain,
                    max_angle_strain=self.max_angle_strain,
                    max_area_mismatch=self.max_area_mismatch,
                )
                matches = zm.run()

                if len(matches) > 0:
                    min_area_match = matches[0]
                    area = min_area_match.area
                    strain = min_area_match.linear_strain
                    misfits[i, j] = np.abs(strain).max()
                    areas[i, j] = area / np.sqrt(substrate[1] * film[1])

        self.misfits = np.round(misfits.T, 8)
        self.areas = areas.T

    def plot_misfits(
        self,
        cmap: str = "magma",
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
            tmp_label = [
                str(i) if i >= 0 else "$\\overline{" + str(-i) + "}$"
                for i in ylabel
            ]
            ylabels.append(f'({"".join(tmp_label)})')

        xlabels = []
        for xlabel in self.substrate_inds:
            tmp_label = [
                str(i) if i >= 0 else "$\\overline{" + str(-i) + "}$"
                for i in xlabel
            ]
            xlabels.append(f'({"".join(tmp_label)})')

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
            film_label = self.film.composition.reduced_formula

        ax.set_ylabel(film_label + " Miller Index", fontsize=fontsize)

        if substrate_label is None:
            substrate_label = self.substrate.composition.reduced_formula

        ax.set_xlabel(substrate_label + " Miller Index", fontsize=fontsize)

        R = 0.85 * s / np.nanmax(s) / 2
        circles = [
            plt.Circle((i, j), radius=r, edgecolor="black", lw=3)
            for r, i, j in zip(R.flat, x.flat, y.flat)
        ]
        col = PatchCollection(
            circles,
            array=c.flatten(),
            cmap=cmap,
            norm=Normalize(
                vmin=np.max([np.nanmin(c) - 0.01 * np.nanmin(c), 0]),
                vmax=np.nanmax(c) + 0.01 * np.nanmax(c),
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
        cbar.set_label("Misfit Percentage", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.ticklabel_format(
            style="sci", scilimits=(-3, 3), useMathText=True
        )
        cbar.ax.yaxis.set_offset_position("left")

        ax.set_aspect("equal")
        fig.tight_layout(pad=0.4)
        fig.savefig(output, bbox_inches="tight")

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
