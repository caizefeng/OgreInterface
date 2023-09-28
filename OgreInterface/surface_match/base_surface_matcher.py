from typing import List, Dict
from copy import deepcopy
import os

from pymatgen.core.structure import Structure
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from scipy.interpolate import RectBivariateSpline, CubicSpline
import numpy as np
from sko.PSO import PSO
from sko.tools import set_run_mode
from tqdm import tqdm

from OgreInterface.surfaces import Interface
from OgreInterface.score_function import (
    generate_input_dict,
    create_batch,
)


def _tqdm_run(self, max_iter=None, precision=None, N=20):
    """
    precision: None or float
        If precision is None, it will run the number of max_iter steps
        If precision is a float, the loop will stop if continuous N difference between pbest less than precision
    N: int
    """
    self.max_iter = max_iter or self.max_iter
    c = 0
    for iter_num in tqdm(range(self.max_iter)):
        self.update_V()
        self.recorder()
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        if precision is not None:
            tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
            if tor_iter < precision:
                c = c + 1
                if c > N:
                    break
            else:
                c = 0
        if self.verbose:
            print(
                "Iter: {}, Best fit: {} at {}".format(
                    iter_num, self.gbest_y, self.gbest_x
                )
            )

        self.gbest_y_hist.append(self.gbest_y)
    self.best_x, self.best_y = self.gbest_x, self.gbest_y
    return self.best_x, self.best_y


PSO.run = _tqdm_run


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
    ):
        self.interface = interface

        self.iface = self.interface.get_interface(orthogonal=True).copy()

        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        self.film_bulk = self.interface.film_oriented_bulk_supercell
        self.sub_bulk = self.interface.substrate_oriented_bulk_supercell

        self.film_bulk.add_site_property(
            "is_film",
            [True] * len(self.film_bulk),
        )
        self.sub_bulk.add_site_property(
            "is_film",
            [False] * len(self.sub_bulk),
        )

        self.film_sc_part = self.interface.get_film_supercell().copy()
        self.sub_sc_part = self.interface.get_substrate_supercell().copy()

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
        adhesion_energies = (
            total_energies - self.film_energy - self.sub_energy
        ) / self.interface.area

        interface_energies = (
            adhesion_energies
            + self.film_surface_energy
            + self.sub_surface_energy
        )

        return adhesion_energies, interface_energies

    def _generate_base_inputs(
        self,
        structure: Structure,
        is_slab: bool = True,
    ):
        inputs = generate_input_dict(
            structure=structure,
            cutoff=self._cutoff + 5.0,
            interface=is_slab,
        )

        return inputs

    def _add_shifts_to_batch(
        self,
        batch_inputs: Dict[str, np.ndarray],
        shifts: np.ndarray,
    ) -> None:
        if "is_film" in batch_inputs:
            n_atoms = batch_inputs["n_atoms"]
            all_shifts = np.repeat(
                shifts.astype(batch_inputs["R"].dtype), repeats=n_atoms, axis=0
            )
            all_shifts[~batch_inputs["is_film"]] *= 0.0
            batch_inputs["R"] += all_shifts
        else:
            raise "_add_shifts_to_batch should only be used on interfaces that have the is_film property"

    def _optimizerPSO(self, func, z_bounds, max_iters, n_particles: int = 25):
        set_run_mode(func, mode="vectorization")
        print("Running 3D Surface Matching with Particle Swarm Optimization:")
        optimizer = PSO(
            func=func,
            pop=n_particles,
            max_iter=max_iters,
            lb=[0.0, 0.0, z_bounds[0]],
            ub=[1.0, 1.0, z_bounds[1]],
            w=0.9,
            c1=0.5,
            c2=0.3,
            verbose=False,
            dim=3,
        )
        optimizer.run()
        cost = optimizer.gbest_y
        pos = optimizer.gbest_x

        return cost, pos

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

        return prim_cart_shifts.reshape(X.shape + (-1,))

    def get_structures_for_DFT(self, output_folder="PES"):
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        all_shifts = self.shifts
        unique_shifts = all_shifts[:-1, :-1]
        shifts = unique_shifts.reshape(-1, 3).dot(self.inv_matrix)

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
            "managua",
        ]
        diverging_names = mpl_diverging_names + cm_diverging_names

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

        ax.contourf(
            X,
            Y,
            Z,
            cmap=cmap,
            levels=200,
            norm=norm,
        )

        if add_color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.1)
            cbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                orientation="horizontal",
            )
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.locator_params(nbins=3)

            if show_max:
                E_opt = np.min(Z)
                label = "$E_{adh}$ (eV) : $E_{min}$ = " + f"{E_opt:.4f} eV"
                cbar.set_label(label, fontsize=fontsize, labelpad=8)
            else:
                label = "$E_{adh}$ (eV)"
                cbar.set_label(label, fontsize=fontsize, labelpad=8)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            ax.tick_params(labelsize=fontsize)

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

    def run_surface_matching(
        self,
        cmap: str = "coolwarm",
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

        energies = []
        for batch_shift in shifts:
            batch_inputs = create_batch(
                inputs=self.iface_inputs,
                batch_size=len(batch_shift),
            )
            self._add_shifts_to_batch(
                batch_inputs=batch_inputs,
                shifts=batch_shift,
            )
            (
                batch_energies,
                _,
                _,
                _,
                _,
            ) = self._calculate(batch_inputs, is_interface=True)
            energies.append(batch_energies)

        interface_energy = np.vstack(energies)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z_adh, Z_iface = self._get_interface_energy(
            total_energies=interface_energy
        )

        if save_raw_data_file is not None:
            if save_raw_data_file.split(".")[-1] != "npz":
                save_raw_data_file = ".".join(
                    save_raw_data_file.split(".")[:-1] + ["npz"]
                )

            np.savez(
                save_raw_data_file,
                x_shifts=X,
                y_shifts=Y,
                energies=Z_adh,
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
            Z=Z_adh,
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
            self._add_shifts_to_batch(
                batch_inputs=inputs, shifts=shift.reshape(1, -1)
            )

            (
                _interface_energy,
                _coulomb,
                _born,
                _,
                _,
            ) = self._calculate(
                inputs,
                is_interface=True,
            )
            interface_energy.append(_interface_energy)
            coulomb.append(_coulomb)
            born.append(_born)

        adhesion_energy, interface_energy = self._get_interface_energy(
            total_energies=interface_energy
        )

        if save_raw_data_file is not None:
            if save_raw_data_file.split(".")[-1] != "npz":
                save_raw_data_file = ".".join(
                    save_raw_data_file.split(".")[:-1] + ["npz"]
                )

            np.savez(
                save_raw_data_file,
                interfacial_distances=interfacial_distances,
                energies=adhesion_energy,
            )

        fig, axs = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        cs = CubicSpline(interfacial_distances, adhesion_energy)
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
            "$E_{adh}$ (eV/$\\AA^{2}$)",
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
        ) = self._calculate(inputs, is_interface=True)

        adhesion_energy, interface_energy = self._get_interface_energy(
            total_energies=total_energy,
        )

        return adhesion_energy[0], interface_energy[0]
