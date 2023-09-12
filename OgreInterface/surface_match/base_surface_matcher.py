from OgreInterface.surfaces import Interface
from OgreInterface.score_function.generate_inputs import (
    generate_input_dict,
    create_batch,
)
from OgreInterface import utils
from typing import List
import numpy as np
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from scipy.interpolate import RectBivariateSpline, CubicSpline
from copy import deepcopy
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
from ase.data import chemical_symbols
import itertools
import time
from pyswarms.single.global_best import GlobalBestPSO
import os


class BaseSurfaceMatcher:
    """Base Class for all other surface matching classes

    The BaseSurfaceMatcher contains all the basic methods to perform surface matching
    that other classes can inherit. This class should not be called on it's own, rather it
    should be used as a building block for other surface matching classes

    Args:
        interface: The Interface object generated using the InterfaceGenerator
        grid_density: The sampling density of the 2D potential energy surface plot (points/Angstrom)
    """

    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
        use_interface_energy: bool = True,
    ):
        self.interface = interface
        self.use_interface_energy = use_interface_energy

        self.iface = self.interface.get_interface(orthogonal=True).copy()

        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        if self.use_interface_energy:
            self.film_part = utils.apply_strain_matrix(
                structure=self.interface.film.oriented_bulk_structure.copy(),
                strain_matrix=self.interface._strain_matrix,
            )
            self.sub_part = (
                self.interface.substrate.oriented_bulk_structure.copy()
            )
            self.film_part.add_site_property(
                "is_film",
                [True] * len(self.film_part),
            )
            self.sub_part.add_site_property(
                "is_film",
                [False] * len(self.sub_part),
            )
        else:
            self.film_part = self.interface.get_film_supercell().copy()
            self.sub_part = self.interface.get_substrate_supercell().copy()

        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.inv_matrix = np.linalg.inv(self.matrix)

        self.grid_density = grid_density

        (
            self.shift_matrix,
            self.shift_images,
        ) = self._get_shift_matrix_and_images()

        self.shifts = self._generate_shifts()

    def _get_interface_energy(self, total_energies: np.ndarray) -> np.ndarray:
        interface_energies = (
            total_energies - self.film_energy - self.sub_energy
        ) / (2 * self.interface.area)

        if self.use_interface_energy:
            return interface_energies
        else:
            return 2 * interface_energies

    def _generate_base_inputs(
        self,
        structure: Structure,
        is_slab: bool = True,
    ):
        inputs = generate_input_dict(
            structure=structure,
            cutoff=self._cutoff,
            interface=is_slab,
        )

        return inputs

    # def _generate_bulk_inputs(self, structure: Structure):
    #     inputs = generate_input_dict(
    #         structure=structure,
    #         cutoff=self._cutoff,
    #         interface=False,
    #     )

    #     return inputs

    def _optimizerPSO(self, func, z_bounds, max_iters, n_particles: int = 15):
        bounds = (
            np.array([0.0, 0.0, z_bounds[0]]),
            np.array([1.0, 1.0, z_bounds[1]]),
        )
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=3,
            options=options,
            bounds=bounds,
        )
        cost, pos = optimizer.optimize(func, iters=max_iters)

        return cost, pos

    def _get_gd_init_points(self):
        sub_struc = self.interface.substrate.oriented_bulk_structure.copy()
        is_top = sub_struc.site_properties["is_top"]
        to_del = np.where(np.logical_not(is_top))[0]
        sub_struc.remove_sites(to_del)

        sub_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self.interface._substrate_a_to_i,
            translation_vec=np.zeros(3),
        )
        sub_struc.apply_operation(sub_a_to_i_op)

        film_struc = self.interface.film.oriented_bulk_structure.copy()
        is_bottom = film_struc.site_properties["is_bottom"]
        to_del = np.where(np.logical_not(is_bottom))[0]
        film_struc.remove_sites(to_del)

        film_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self.interface._film_a_to_i,
            translation_vec=np.zeros(3),
        )
        film_struc.apply_operation(film_a_to_i_op)

        unstrained_film_matrix = film_struc.lattice.matrix
        strain_matrix = (
            self.interface._film_supercell.lattice.inv_matrix
            @ self.interface._strained_sub.lattice.matrix
        )
        strain_matrix[-1] = np.array([0, 0, 1])
        strained_matrix = unstrained_film_matrix.dot(strain_matrix.T)
        film_struc = Structure(
            lattice=Lattice(strained_matrix),
            species=film_struc.species,
            coords=film_struc.frac_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=film_struc.site_properties,
        )

        # Poscar(sub_struc).write_file("POSCAR_sub_top")
        # Poscar(film_struc).write_file("POSCAR_film_bot")

        sub_equivs = sub_struc.site_properties["bulk_equivalent"]
        film_equivs = film_struc.site_properties["bulk_equivalent"]

        _, sub_unique = np.unique(sub_equivs, return_index=True)
        _, film_unique = np.unique(film_equivs, return_index=True)

        film_cart_coords = film_struc.cart_coords[:, :2]
        sub_cart_coords = sub_struc.cart_coords[:, :2]
        # inds = itertools.product(
        #     # film_unique,
        #     # sub_unique,
        #     # [1, 4]
        #     range(len(film_cart_coords)),
        #     range(len(sub_cart_coords)),
        # )

        shifts = []
        unique_inds = []
        for i, film_coords in enumerate(film_cart_coords):
            for j, sub_coords in enumerate(sub_cart_coords):
                shift = sub_coords - film_coords
                shifts.append(shift)
                unique_inds.append((film_equivs[i], sub_equivs[j]))
                # unique_inds.append(film_equivs[i])

        shifts = np.c_[shifts, np.zeros(len(shifts))]
        inv_matrix = np.linalg.inv(self.shift_matrix)
        frac_shifts = shifts.dot(inv_matrix)
        frac_shifts = np.round(np.mod(frac_shifts, 1), 5)
        # unique_shifts = np.unique(frac_shifts, axis=0)
        points = (frac_shifts + self.shift_images[0]).dot(self.shift_matrix)

        colors = [
            "red",
            "green",
            "blue",
            "magenta",
            "orange",
            "purple",
            "white",
            "yellow",
        ]
        color_dict = {s: c for s, c in zip(set(unique_inds), colors)}
        plot_colors = [color_dict[s] for s in unique_inds]

        return points[:, :2]
        # , plot_colors

    def _get_shift_matrix_and_images(self) -> List[np.ndarray]:
        (
            sub_matrix,
            sub_images,
            film_matrix,
            film_images,
        ) = self.interface._get_oriented_cell_and_images(strain=True)

        if self.interface.substrate.area < self.interface.film.area:
            shift_matrix = sub_matrix
            shift_images = sub_images
        else:
            shift_matrix = film_matrix
            shift_images = film_images

        return shift_matrix, shift_images

    def _generate_shifts(self) -> List[np.ndarray]:
        iface_inv_matrix = (
            self.interface._orthogonal_structure.lattice.inv_matrix
        )

        grid_density_x = int(
            np.round(np.linalg.norm(self.shift_matrix[0]) * self.grid_density)
        )
        grid_density_y = int(
            np.round(np.linalg.norm(self.shift_matrix[1]) * self.grid_density)
        )

        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y

        grid_x = np.linspace(0, 1, grid_density_x)
        grid_y = np.linspace(0, 1, grid_density_y)

        X, Y = np.meshgrid(grid_x, grid_y)
        self.X_shape = X.shape

        prim_frac_shifts = (
            np.c_[X.ravel(), Y.ravel(), np.zeros(Y.shape).ravel()]
            + self.shift_images[0]
        )

        prim_cart_shifts = prim_frac_shifts.dot(self.shift_matrix)
        # iface_frac_shifts = prim_cart_shifts.dot(iface_inv_matrix).reshape(
        #     X.shape + (-1,)
        # )

        return prim_cart_shifts.reshape(X.shape + (-1,))

    # TODO create a function that can write out the shifted structures for DFT calculations and also read in the structure and plot them nicely
    def get_structures_for_DFT(self, output_folder="PES"):
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        all_shifts = self.shifts
        unique_shifts = all_shifts[:-1, :-1]
        shifts = unique_shifts.reshape(-1, 3).dot(self.inv_matrix)
        # shifts = np.mod(shifts, 1)

        for i, shift in enumerate(shifts):
            self.interface.shift_film_inplane(
                x_shift=shift[0],
                y_shift=shift[1],
                fractional=True,
            )
            self.interface.write_file(
                output=os.path.join(output_folder, f"POSCAR_{i:04d}")
            )
            self.interface.shift_film_inplane(
                x_shift=-shift[0],
                y_shift=-shift[1],
                fractional=True,
            )

    def get_structures_for_DFT_z_shift(
        self,
        interfacial_distances: np.ndarray,
        output_folder: str = "z_shift",
    ) -> None:
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        for i, dist in enumerate(interfacial_distances):
            self.interface.set_interfacial_distance(interfacial_distance=dist)
            self.interface.write_file(
                output=os.path.join(output_folder, f"POSCAR_{i:04d}")
            )

    def plot_DFT_data(
        self,
        energies: np.ndarray,
        sub_energy: float = 0.0,
        film_energy: float = 0.0,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        dpi: int = 400,
        show_opt: bool = False,
    ) -> float:
        """This function plots the 2D potential energy surface (PES) from DFT (or other) calculations

        Args:
            energies: Numpy array of the DFT energies in the same order as the output of the get_structures_for_DFT() function
            sub_energy: Total energy of the substrate supercell section of the interface (include this for adhesion energy)
            film_energy: Total energy of the film supercell section of the interface (include this for adhesion energy)
            cmap: The colormap to use for the PES, any matplotlib compatible color map will work
            fontsize: Fontsize of all the plot labels
            output: Output file name
            dpi: Resolution of the figure (dots per inch)
            show_opt: Determines if the optimal value is printed on the figure


        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        init_shape = (self.X_shape[0] - 1, self.X_shape[1] - 1)
        unique_energies = energies.reshape(init_shape)
        interface_energy = np.c_[unique_energies, unique_energies[:, 0]]
        interface_energy = np.vstack([interface_energy, interface_energy[0]])

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        N_sub_layers = self.interface.substrate.layers
        N_film_layers = self.interface.film.layers
        N_sub_sc = np.linalg.det(self.interface.match.substrate_sl_transform)
        N_film_sc = np.linalg.det(self.interface.match.film_sl_transform)

        sub_obs = sub_energy * N_sub_sc * N_sub_layers
        film_obs = film_energy * N_film_sc * N_film_layers

        Z = (interface_energy - film_obs - sub_obs) / (2 * self.interface.area)

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]

        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()

        ratio = y_size / x_size

        if ratio < 1:
            figx = 5 / ratio
            figy = 5
        else:
            figx = 5
            figy = 5 * ratio

        fig, ax = plt.subplots(
            figsize=(figx, figy),
            dpi=dpi,
        )

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=1,
            zorder=300,
        )

        max_Z = self._plot_surface_matching(
            fig=fig,
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            dpi=dpi,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_opt,
            shift=True,
        )

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z

    def plot_DFT_z_shift(
        self,
        interfacial_distances: np.ndarray,
        energies: np.ndarray,
        film_energy: float = 0.0,
        sub_energy: float = 0.0,
        figsize: tuple = (4, 3),
        fontsize: int = 12,
        output: str = "z_shift.png",
        dpi: int = 400,
    ):
        """This function calculates the negated adhesion energy of an interface as a function of the interfacial distance

        Args:
            interfacial_distances: numpy array of the interfacial distances that should be calculated
            figsize: Size of the figure in inches (x_size, y_size)
            fontsize: Fontsize of all the plot labels
            output: Output file name
            dpi: Resolution of the figure (dots per inch)

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        N_sub_layers = self.interface.substrate.layers
        N_film_layers = self.interface.film.layers
        N_sub_sc = np.linalg.det(self.interface.match.substrate_sl_transform)
        N_film_sc = np.linalg.det(self.interface.match.film_sl_transform)

        sub_obs = sub_energy * N_sub_sc * N_sub_layers
        film_obs = film_energy * N_film_sc * N_film_layers

        interface_energy = (energies - film_obs - sub_obs) / (
            2 * self.interface.area
        )

        fig, axs = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        cs = CubicSpline(interfacial_distances, interface_energy)
        new_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        new_y = cs(new_x)

        opt_d = new_x[np.argmin(new_y)]
        opt_E = np.min(new_y)
        self.opt_d_interface = opt_d

        axs.annotate(
            "$d_{int}^{opt}$" + f" $= {opt_d:.3f}$",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="black",
            ),
        )

        axs.plot(
            new_x,
            new_y,
            color="black",
            linewidth=1,
        )
        axs.scatter(
            [opt_d],
            [opt_E],
            color="black",
            marker="x",
        )
        axs.tick_params(labelsize=fontsize)
        axs.set_ylabel(
            "$-E_{adh}$ (eV/$\\AA^{2}$)",
            fontsize=fontsize,
        )
        axs.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return opt_E

    def get_cart_xy_shifts(self, ab):
        frac_abc = np.c_[ab, np.zeros(len(ab))]
        cart_xyz = (frac_abc + self.shift_images[0]).dot(self.shift_matrix)

        return cart_xyz[:, :2]

    def get_frac_xy_shifts(self, xy):
        cart_xyz = np.c_[xy, np.zeros(len(xy))]
        inv_shift = np.linalg.inv(self.shift_matrix)
        frac_abc = cart_xyz.dot(inv_shift)
        frac_abc = np.mod(frac_abc, 1)

        return frac_abc[:, :2]

    def _plot_heatmap(
        self, fig, ax, X, Y, Z, cmap, fontsize, show_max, add_color_bar
    ):
        ax.set_xlabel(r"Shift in $x$ ($\AA$)", fontsize=fontsize)
        ax.set_ylabel(r"Shift in $y$ ($\AA$)", fontsize=fontsize)

        mpl_diverging_names = [
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ]
        cm_diverging_names = [
            "broc",
            "cork",
            "vik",
            "lisbon",
            "tofino",
            "berlin",
            "roma",
            "bam",
            "vanimo",
        ]
        diverging_names = mpl_diverging_names + cm_diverging_names

        Z /= np.abs(np.min(Z))

        min_Z = np.nanmin(Z)
        max_Z = np.nanmax(Z)
        if type(cmap) == str:
            if cmap in diverging_names:
                bound = np.max([np.abs(min_Z), np.abs(max_Z)])
                norm = Normalize(vmin=-bound, vmax=bound)
            else:
                norm = Normalize(vmin=min_Z, vmax=max_Z)
        elif type(cmap) == ListedColormap:
            name = cmap.name
            if name in diverging_names:
                bound = np.max([np.abs(min_Z), np.abs(max_Z)])
                norm = Normalize(vmin=-bound, vmax=bound)
            else:
                norm = Normalize(vmin=min_Z, vmax=max_Z)
        else:
            norm = Normalize(vmin=min_Z, vmax=max_Z)

        im = ax.contourf(
            X,
            Y,
            Z,
            cmap=cmap,
            levels=200,
            # norm=Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z)),
            norm=norm,
        )

        if add_color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.locator_params(nbins=3)

            if show_max:
                E_max = np.min(Z)
                label = "$-E_{adh}/|E_{min}|$ : $E_{min}$ = " + f"{E_max:.4f}"
                # label = (
                #     "$-E_{adh}$ (eV/$\\AA^{2}$) : "
                #     + "$E_{min}$ = "
                #     + f"{E_max:.4f}"
                # )
                cbar.set_label(label, fontsize=fontsize, labelpad=8)
            else:
                # label = "$-E_{adh}$ (eV/$\\AA^{2}$)"
                label = "$-E_{adh}/|E_{min}|$"
                cbar.set_label(label, fontsize=fontsize, labelpad=8)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            ax.tick_params(labelsize=fontsize)

    # def _get_interpolated_data_old(self, Z, image):
    #     print("Using old interpolation")
    #     x_grid = np.linspace(0, 1, self.grid_density_x)
    #     y_grid = np.linspace(0, 1, self.grid_density_y)
    #     spline = RectBivariateSpline(y_grid, x_grid, Z)

    #     x_grid_interp = np.linspace(0, 1, 101)
    #     y_grid_interp = np.linspace(0, 1, 101)

    #     X_interp, Y_interp = np.meshgrid(x_grid_interp, y_grid_interp)
    #     Z_interp = spline.ev(xi=Y_interp, yi=X_interp)
    #     frac_shifts = (
    #         np.c_[
    #             X_interp.ravel(),
    #             Y_interp.ravel(),
    #             np.zeros(X_interp.shape).ravel(),
    #         ]
    #         + image
    #     )

    #     cart_shifts = frac_shifts.dot(self.shift_matrix)

    #     X_cart = cart_shifts[:, 0].reshape(X_interp.shape)
    #     Y_cart = cart_shifts[:, 1].reshape(Y_interp.shape)

    #     return X_cart, Y_cart, Z_interp

    def _get_interpolated_data(self, Z, image):
        x_grid = np.linspace(-1, 2, (3 * self.grid_density_x) - 2)
        y_grid = np.linspace(-1, 2, (3 * self.grid_density_y) - 2)
        Z_horiz = np.c_[Z, Z[:, 1:-1], Z]
        Z_periodic = np.r_[Z_horiz, Z_horiz[1:-1, :], Z_horiz]
        spline = RectBivariateSpline(y_grid, x_grid, Z_periodic)

        x_grid_interp = np.linspace(0, 1, 101)
        y_grid_interp = np.linspace(0, 1, 101)

        X_interp, Y_interp = np.meshgrid(x_grid_interp, y_grid_interp)
        Z_interp = spline.ev(xi=Y_interp, yi=X_interp)
        frac_shifts = (
            np.c_[
                X_interp.ravel(),
                Y_interp.ravel(),
                np.zeros(X_interp.shape).ravel(),
            ]
            + image
        )

        cart_shifts = frac_shifts.dot(self.shift_matrix)

        X_cart = cart_shifts[:, 0].reshape(X_interp.shape)
        Y_cart = cart_shifts[:, 1].reshape(Y_interp.shape)

        return X_cart, Y_cart, Z_interp

    def _plot_surface_matching(
        self,
        fig,
        ax,
        X,
        Y,
        Z,
        dpi,
        cmap,
        fontsize,
        show_max,
        shift,
    ):
        for i, image in enumerate(self.shift_images):
            X_plot, Y_plot, Z_plot = self._get_interpolated_data(Z, image)

            if i == 0:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=True,
                )

                frac_shifts = np.c_[
                    X_plot.ravel(),
                    Y_plot.ravel(),
                    np.zeros(Y_plot.shape).ravel(),
                ].dot(np.linalg.inv(self.matrix))

                opt_shift = frac_shifts[np.argmin(Z_plot.ravel())]
                opt_shift = np.mod(opt_shift, 1)
                max_Z = np.min(Z_plot)
                plot_shift = opt_shift.dot(self.matrix)

                ax.scatter(
                    [plot_shift[0]],
                    [plot_shift[1]],
                    fc="white",
                    ec="black",
                    marker="X",
                    s=100,
                    zorder=10,
                )

                if shift:
                    self.opt_xy_shift = opt_shift[:2]
            else:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=False,
                )

            coords = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            )

            sc_shifts = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                ]
            )

            for shift in sc_shifts:
                shift_coords = (coords + shift).dot(self.matrix)
                poly = Polygon(
                    xy=shift_coords[:, :2],
                    closed=True,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=1,
                    zorder=200,
                )
                ax.add_patch(poly)

        return max_Z

    def _adam(
        self,
        score_func,
        score_func_inputs,
        beta1=0.9,
        beta2=0.999,
        eta=0.01,
        epsilon=1e-7,
        iterations=300,
    ):
        inv_shift_matrix = np.linalg.inv(self.shift_matrix)
        init_position = score_func_inputs["shift"]
        opt_position = [np.copy(init_position)]
        m = np.zeros(init_position.shape)
        v = np.zeros(init_position.shape)

        for i in range(iterations):
            print(opt_position[i])
            force_norm, gradient = score_func.forward(**score_func_inputs)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)
            update = m_hat / (np.sqrt(v_hat) + epsilon)
            new_opt_position = opt_position[i] - eta * update

            new_opt_frac_coords = (
                np.array([new_opt_position[0], new_opt_position[1], 0.0]).dot(
                    inv_shift_matrix
                )
                - self.shift_images[0]
            )
            new_opt_frac_coords = np.mod(new_opt_frac_coords, 1)
            new_opt_cart_coords = (
                new_opt_frac_coords + self.shift_images[0]
            ).dot(self.shift_matrix)
            new_opt_position[:2] = new_opt_cart_coords[:2]
            opt_position.append(new_opt_position)
            score_func_inputs["shift"] = torch.from_numpy(new_opt_position)

        opt_position = np.vstack(opt_position)

        return opt_position

    def run_surface_matching(
        self,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        dpi: int = 400,
        show_opt: bool = False,
        save_raw_data_file=None,
    ) -> float:
        """This function calculates the 2D potential energy surface (PES)

        Args:
            cmap: The colormap to use for the PES, any matplotlib compatible color map will work
            fontsize: Fontsize of all the plot labels
            output: Output file name
            dpi: Resolution of the figure (dots per inch)
            show_opt: Determines if the optimal value is printed on the figure
            save_raw_data_file: If you put a valid file path (i.e. anything ending with .npz) then the
                raw data will be saved there. It can be loaded in via data = np.load(save_raw_data_file)
                and the data is: x_shifts = data["x_shifts"], y_shifts = data["y_shifts"], energies = data["energies"]

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        shifts = self.shifts

        sub_inputs = create_batch(inputs=self.sub_inputs, batch_size=1)
        film_inputs = create_batch(inputs=self.film_inputs, batch_size=1)

        sub_energy, _, _, _, _ = self._calculate(
            sub_inputs,
            shifts=np.zeros((1, 3)),
        )
        film_energy, _, _, _, _ = self._calculate(
            film_inputs,
            shifts=np.zeros((1, 3)),
        )

        energies = []
        grads = []
        for batch_shift in shifts:
            batch_inputs = create_batch(
                inputs=self.iface_inputs,
                batch_size=len(batch_shift),
            )
            (
                batch_energies,
                _,
                _,
                batch_grads,
                _,
            ) = self._calculate(batch_inputs, shifts=batch_shift)
            energies.append(batch_energies)
            grads.append(batch_grads)

        interface_energy = np.vstack(energies)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z = self._get_interface_energy(total_energies=interface_energy)

        if save_raw_data_file is not None:
            if save_raw_data_file.split(".")[-1] != "npz":
                save_raw_data_file = ".".join(
                    save_raw_data_file.split(".")[:-1] + ["npz"]
                )

            np.savez(
                save_raw_data_file,
                x_shifts=X,
                y_shifts=Y,
                energies=Z,
            )

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]

        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()

        ratio = y_size / x_size

        if ratio < 1:
            figx = 5 / ratio
            figy = 5
        else:
            figx = 5
            figy = 5 * ratio

        fig, ax = plt.subplots(
            figsize=(figx, figy),
            dpi=dpi,
        )

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=1,
            zorder=300,
        )

        max_Z = self._plot_surface_matching(
            fig=fig,
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            dpi=dpi,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_opt,
            shift=True,
        )

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z

    def run_z_shift(
        self,
        interfacial_distances: np.ndarray,
        figsize: tuple = (4, 3),
        fontsize: int = 12,
        output: str = "z_shift.png",
        dpi: int = 400,
        save_raw_data_file=None,
    ):
        """This function calculates the negated adhesion energy of an interface as a function of the interfacial distance

        Args:
            interfacial_distances: numpy array of the interfacial distances that should be calculated
            figsize: Size of the figure in inches (x_size, y_size)
            fontsize: Fontsize of all the plot labels
            output: Output file name
            dpi: Resolution of the figure (dots per inch)
            save_raw_data_file: If you put a valid file path (i.e. anything ending with .npz) then the
                raw data will be saved there. It can be loaded in via data = np.load(save_raw_data_file)
                and the data is: interfacial_distances = data["interfacial_distances"], energies = data["energies"]

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        zeros = np.zeros(len(interfacial_distances))
        shifts = np.c_[zeros, zeros, interfacial_distances - self.d_interface]

        interface_energy = []
        coulomb = []
        born = []
        for shift in shifts:
            inputs = create_batch(self.iface_inputs, batch_size=1)

            (
                _interface_energy,
                _coulomb,
                _born,
                _,
                _,
            ) = self._calculate(inputs, shifts=shift.reshape(1, -1))
            interface_energy.append(_interface_energy)
            coulomb.append(_coulomb)
            born.append(_born)

        interface_energy = self._get_interface_energy(
            total_energies=interface_energy
        )
        # interface_energy = (
        #     -(
        #         (self.film_energy + self.sub_energy)
        #         - np.array(interface_energy)
        #     )
        #     / self.interface.area
        # )

        if save_raw_data_file is not None:
            if save_raw_data_file.split(".")[-1] != "npz":
                save_raw_data_file = ".".join(
                    save_raw_data_file.split(".")[:-1] + ["npz"]
                )

            np.savez(
                save_raw_data_file,
                interfacial_distances=interfacial_distances,
                energies=interface_energy,
            )

        fig, axs = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        cs = CubicSpline(interfacial_distances, interface_energy)
        new_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        new_y = cs(new_x)

        opt_d = new_x[np.argmin(new_y)]
        opt_E = np.min(new_y)
        self.opt_d_interface = opt_d

        axs.annotate(
            "$d_{int}^{opt}$" + f" $= {opt_d:.3f}$",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="black",
            ),
        )

        axs.plot(
            new_x,
            new_y,
            color="black",
            linewidth=1,
        )
        axs.scatter(
            [opt_d],
            [opt_E],
            color="black",
            marker="x",
        )
        axs.tick_params(labelsize=fontsize)
        axs.set_ylabel(
            "$-E_{adh}$ (eV/$\\AA^{2}$)",
            fontsize=fontsize,
        )
        axs.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return opt_E

    def get_current_energy(
        self,
    ):
        """This function calculates the energy of the current interface structure

        Returns:
            Interface or Adhesion energy of the interface
        """
        inputs = create_batch(self.iface_inputs, batch_size=1)

        (
            total_energy,
            _,
            _,
            _,
            _,
        ) = self._calculate(inputs, shifts=np.zeros((1, 3)))

        interface_energy = self._get_interface_energy(
            total_energies=total_energy,
        )

        return interface_energy[0]
