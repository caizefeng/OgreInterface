import tpying as tp
import copy
import os
from abc import ABC, abstractmethod, abstractproperty

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
from OgreInterface import utils

# from OgreInterface.score_function import (
#     generate_input_dict,
#     create_batch,
# )


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


class BaseSurfaceMatcher(ABC):
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
        # Interface or Molecular Interface Object
        self.interface = interface

        # Actual Structure of the interface
        self.iface = self.interface.get_interface(orthogonal=True).copy()

        # If passivated remove the pseudo hydrogens
        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        # Get the strained film oriented bulk structure
        self.film_obs = self.interface.film_oriented_bulk_structure

        # Get the strained substrate oriented bulk structure
        self.sub_obs = self.interface.substrate_oriented_bulk_structure

        # Get the film supercell of the interface
        self.film_supercell = self.interface.get_film_supercell().copy()

        # If passivated remove the pseudo hydrogens
        if self.interface.film._passivated:
            H_inds = np.where(
                np.array(self.film_supercell.atomic_numbers) == 1
            )[0]
            self.film_supercell.remove_sites(H_inds)

        # Get the substrate supercell of the interface
        self.sub_supercell = self.interface.get_substrate_supercell().copy()

        # If passivated remove the pseudo hydrogens
        if self.interface.substrate._passivated:
            H_inds = np.where(
                np.array(self.sub_supercell.atomic_numbers) == 1
            )[0]
            self.sub_supercell.remove_sites(H_inds)

        # Dictionary of constant structures
        self.constant_structure_dict = {
            "film_supercell": self.film_supercell,
            "sub_supercell": self.sub_supercell,
        }

        # Dictionary of the total energies of the constant structures
        (
            self.film_supercell_energy,
            self.sub_supercell_energy,
        ) = self.precalculate_supercell_energies()

        (
            self.film_surface_energy,
            self.sub_supercell_energy,
        ) = self.precalculate_surface_energies()

        # Get the lattice matrix of the interface
        self.matrix = copy.deepcopy(
            interface._orthogonal_structure.lattice.matrix
        )

        # Get the volume of the matrix
        self._vol = np.linalg.det(self.matrix)

        # If the volume is negative (shouldn't happen ever anymore) this means
        # the interface has a left handed basis so we will change it to a right
        # handed basis
        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        # Get the inverse matrix (used for getting the unique shifts)
        self.inv_matrix = np.linalg.inv(self.matrix)

        # Grid density of the 2D PES
        self.grid_density = grid_density

        # Get the matrix used to determine the shifts (smallest surface area)
        self.shift_matrix = self._get_shift_matrix()

        # Generate the shifts for the 2D PES
        self.shifts = self._generate_shifts()

    @abstractmethod
    def generate_constant_inputs(
        self,
        structure: Structure,
    ):
        """
        This method is used to generate the inputs of the calculate function
        for the structures that will stay constant throughout the surface
        matching process (i.e. OBS, supercells)
        """
        pass

    @abstractmethod
    def generate_interface_inputs(
        self,
        structure: Structure,
        shifts: np.ndarray,
    ):
        """
        This method is used to generate the inputs of the calculate function
        for the interface given various shifts
        """
        pass

    @abstractmethod
    def calculate(self, inupts):
        """
        This method is used to calculate the total energy of the structure with
        the given method of calculating the energy (i.e. DFT, ML-potential)
        """
        pass

    def precalculate_supercell_energies(self) -> tp.Tuple[float, float]:
        """
        This method is used the calculate the energies of the structures that
        will stay constant throughout the surface matching process
        """
        film_inputs = self.generate_constant_inputs(
            structure=self.film_supercell
        )
        film_total_energy = self.calculate(film_inputs)[0]

        sub_inputs = self.generate_constant_inputs(
            structure=self.sub_supercell
        )
        sub_total_energy = self.calculate(sub_inputs)[0]

        return film_total_energy, sub_total_energy

    def precalculate_surface_energies(self) -> tp.Tuple[float, float]:
        """
        This method is used to calculate the surface/cleavage energy of the
        film and substrate
        """
        film_surface_energy = self._get_surface_energy(
            oriented_bulk_structure=self.film_obs,
            layers=self.interface.film.layers,
        )
        sub_surface_energy = self._get_surface_energy(
            oriented_bulk_structure=self.sub_obs,
            layers=self.interface.substrate.layers,
        )

        return film_surface_energy, sub_surface_energy

    def _get_surface_energy(
        self,
        oriented_bulk_structure: Structure,
        layers: int,
    ) -> float:
        # Lattice matrix of the OBS structure
        matrix = oriented_bulk_structure.lattice.matrix

        # Area of the OBS structure (slab cross section)
        surface_area = np.linalg.norm(np.cross(matrix[0], matrix[1]))

        # Slab generated from the OBS
        slab = utils.get_layer_supercell(
            structure=oriented_bulk_structure,
            layers=layers,
        )

        # Self interface generated from the OBS
        double_slab = utils.get_layer_supercell(
            structure=oriented_bulk_structure,
            layers=2 * layers,
        )

        # Layer index property
        double_slab_layers = np.array(
            self.double_slab.site_properties["layer_index"]
        )

        # Is film site properties
        is_film = (double_slab_layers >= layers).astype(bool)

        # Add the is_film property to the self interface
        double_slab.add_site_property(
            "is_film",
            is_film.tolist(),
        )

        # Get the default interfacial distance
        top_sub = double_slab.cart_coords[~is_film][:, -1].max()
        bot_film = double_slab.cart_coords[is_film][:, -1].min()
        default_distance = bot_film - top_sub

        interfacial_distances = np.linspace(
            0.5 * default_distance,
            2.0 * default_distance,
            21,
        )

        zeros = np.zeros(len(interfacial_distances))
        shifts = np.c_[zeros, zeros, interfacial_distances - default_distance]

        double_slab_inputs = self.generate_interface_inputs(
            structure=double_slab,
            shifts=shifts,
        )
        double_slab_energies = self.calculate(inputs=double_slab_inputs)

        slab_inputs = self.generate_constant_inputs(structure=slab)
        slab_energy = self.calculate(inputs=slab_inputs)[0]

        cleavage_energy = (double_slab_energies - (2 * slab_energy)) / (
            2 * surface_area
        )

        cs = CubicSpline(interfacial_distances, cleavage_energy)

        interp_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        interp_y = cs(interp_x)

        opt_E = -np.min(interp_y)

        return opt_E

    def get_adhesion_energy(
        self,
        total_energies: np.ndarray,
    ) -> np.ndarray:
        adhesion_energies = (
            total_energies
            - self.film_supercell_energy
            - self.sub_supercell_energy
        ) / self.interface.area

        return adhesion_energies

    def get_interface_energy(
        self,
        adhesion_energies: np.ndarray,
    ) -> np.ndarray:
        interface_energies = (
            adhesion_energies
            + self.film_surface_energy
            + self.sub_surface_energy
        )

        return interface_energies

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

    def _get_shift_matrix(self) -> np.ndarray:
        if self.interface.substrate.area < self.interface.film.area:
            return copy.deepcopy(self.sub_obs.lattice.matrix)
        else:
            return copy.deepcopy(self.film_obs.lattice.matrix)

    def _generate_shifts(self) -> tp.List[np.ndarray]:
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

        prim_frac_shifts = np.c_[
            X.ravel(),
            Y.ravel(),
            np.zeros(Y.shape).ravel(),
        ]

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

    def _get_figure_for_PES(
        self,
        padding: float,
        dpi: int,
    ):
        min_xy = ((-1 * padding) * np.ones(2)).dot(self.matrix[:2])
        max_xy = ((1 + padding) * np.ones(2)).dot(self.matrix[:2])

        square_length = (max_xy - min_xy).max()
        square_length = np.abs(max_xy - min_xy).max()

        fig, ax = plt.subplots(
            figsize=(5, 5),
            dpi=dpi,
        )

        ax.set_xlim(-square_length / 2, square_length / 2)
        ax.set_ylim(-square_length / 2, square_length / 2)

        return fig, ax, square_length

    def plot_DFT_data(
        self,
        energies: np.ndarray,
        sub_energy: float = 0.0,
        film_energy: float = 0.0,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        dpi: int = 400,
        show_opt_energy: bool = False,
        show_opt_shift: bool = True,
        scale_data: bool = False,
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

        Z = (interface_energy - sub_energy - film_energy) / self.interface.area

        # if scale_data:
        #     Z /= max(abs(Z.min()), abs(Z.max()))

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
            show_max=show_opt_energy,
            show_shift=show_opt_shift,
            scale_data=scale_data,
            shift=True,
        )

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", transparent=False)
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
        interface_energy = (energies - film_energy - sub_energy) / (
            self.interface.area
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
        fig.savefig(output, bbox_inches="tight", transparent=False)
        plt.close(fig)

        return opt_E

    def get_cart_xy_shifts(self, ab):
        frac_abc = np.c_[ab, np.zeros(len(ab))]
        cart_xyz = frac_abc.dot(self.shift_matrix)

        return cart_xyz[:, :2]

    def get_frac_xy_shifts(self, xy):
        cart_xyz = np.c_[xy, np.zeros(len(xy))]
        inv_shift = np.linalg.inv(self.shift_matrix)
        frac_abc = cart_xyz.dot(inv_shift)
        frac_abc = np.mod(frac_abc, 1)

        return frac_abc[:, :2]

    def _plot_heatmap(
        self,
        fig,
        ax,
        X,
        Y,
        Z,
        cmap,
        fontsize,
        show_max,
        scale_data,
        add_color_bar,
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
        if type(cmap) is str:
            if cmap in diverging_names:
                bound = np.max([np.abs(min_Z), np.abs(max_Z)])
                norm = Normalize(vmin=-bound, vmax=bound)
            else:
                norm = Normalize(vmin=min_Z, vmax=max_Z)
        elif type(cmap) is ListedColormap:
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

            if scale_data:
                units = ""
                base_label = "$E_{adh}$/max(|$E_{adh}$|)"
            else:
                units = " (eV/$\\AA^{2}$)"
                base_label = "$E_{adh}$" + units

            if show_max:
                E_opt = np.min(Z)
                label = base_label + " : $E_{min}$ = " + f"{E_opt:.4f}" + units
                cbar.set_label(label, fontsize=fontsize, labelpad=8)
            else:
                label = base_label
                cbar.set_label(label, fontsize=fontsize, labelpad=8)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cax.xaxis.set_ticks(
                [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax],
                [
                    f"{norm.vmin:.2f}",
                    f"{(norm.vmin + norm.vmax) / 2:.2f}",
                    f"{norm.vmax:.2f}",
                ],
            )
            ax.tick_params(labelsize=fontsize)

    def _evaluate_spline(
        self,
        spline: RectBivariateSpline,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            spline: RectBivariateSpline of the PES surface
            X: Grid data to plot on the full PES
            Y: Grid data to plot on the full PES

        Returns:
            Z data for the full PES
        """
        cart_points = np.c_[X.ravel(), Y.ravel(), np.zeros(X.shape).ravel()]
        frac_points = cart_points.dot(np.linalg.inv(self.shift_matrix))
        mod_frac_points = np.mod(frac_points, 1.0)

        X_frac = mod_frac_points[:, 0].reshape(X.shape)
        Y_frac = mod_frac_points[:, 1].reshape(Y.shape)

        return spline.ev(xi=Y_frac, yi=X_frac)

    def _get_spline(self, Z: np.ndarray) -> RectBivariateSpline:
        x_grid = np.linspace(-1, 2, (3 * self.grid_density_x) - 2)
        y_grid = np.linspace(-1, 2, (3 * self.grid_density_y) - 2)
        Z_horiz = np.c_[Z, Z[:, 1:-1], Z]
        Z_periodic = np.r_[Z_horiz, Z_horiz[1:-1, :], Z_horiz]
        spline = RectBivariateSpline(y_grid, x_grid, Z_periodic)

        return spline

    def _plot_surface_matching(
        self,
        fig,
        ax,
        X_plot,
        Y_plot,
        Z,
        dpi,
        cmap,
        fontsize,
        show_max,
        show_shift,
        scale_data,
        shift,
    ):
        spline = self._get_spline(Z=Z)
        Z_plot = self._evaluate_spline(
            spline=spline,
            X=X_plot,
            Y=Y_plot,
        )
        self._plot_heatmap(
            fig=fig,
            ax=ax,
            X=X_plot,
            Y=Y_plot,
            Z=Z_plot,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_max,
            scale_data=scale_data,
            add_color_bar=True,
        )

        return np.max(Z_plot)

    def run_surface_matching(
        self,
        cmap: str = "coolwarm",
        fontsize: int = 14,
        output: str = "PES.png",
        dpi: int = 400,
        show_opt_energy: bool = False,
        show_opt_shift: bool = True,
        scale_data: bool = False,
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

        total_energies = []

        for batch_shift in shifts:
            batch_inputs = self.generate_interface_inputs(
                structure=self.iface,
                shifts=batch_shift,
            )
            batch_total_energies = self.calculate(inputs=batch_inputs)
            total_energies.append(batch_total_energies)

        total_energies = np.vstack(total_energies)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z_adh = self.get_adhesion_energy(total_energies=total_energies)

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
        borders -= (a + b) / 2

        fig, ax, square_length = self._get_figure_for_PES(padding=0.1, dpi=dpi)
        grid = np.linspace(-square_length / 2, square_length / 2, 501)

        X_plot, Y_plot = np.meshgrid(grid, grid)

        max_Z = self._plot_surface_matching(
            fig=fig,
            ax=ax,
            X_plot=X_plot,
            Y_plot=Y_plot,
            Z=Z_adh,
            dpi=dpi,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_opt_energy,
            show_shift=show_opt_shift,
            scale_data=scale_data,
            shift=True,
        )

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=1,
            zorder=300,
        )

        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", transparent=False)
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
        batch_shifts = np.array_split(shifts, len(shifts) // 10, axis=0)

        total_energies = []
        for batch_shift in batch_shifts:
            batch_inputs = self.generate_interface_inputs(
                structure=self.iface_inputs,
                shifts=batch_shift.reshape(1, -1),
            )
            batch_total_energies = self.calculate(inputs=batch_inputs)
            total_energies.append(batch_total_energies)

        total_energies = np.concatenate(total_energies)

        adhesion_energies = self.get_adhesion_energy(
            total_energies=total_energies
        )

        if save_raw_data_file is not None:
            if save_raw_data_file.split(".")[-1] != "npz":
                save_raw_data_file = ".".join(
                    save_raw_data_file.split(".")[:-1] + ["npz"]
                )

            np.savez(
                save_raw_data_file,
                interfacial_distances=interfacial_distances,
                adhesion_energies=adhesion_energies,
            )

        fig, axs = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        cs = CubicSpline(interfacial_distances, adhesion_energies)
        interp_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        interp_y = cs(interp_x)

        opt_d = interp_x[np.argmin(interp_y)]
        opt_E = np.min(interp_y)

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
            interp_x,
            interp_y,
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
        fig.savefig(output, bbox_inches="tight", transparent=False)
        plt.close(fig)

        return opt_E

    def get_current_energy(
        self,
    ):
        """This function calculates the energy of the current interface structure

        Returns:
            Interface or Adhesion energy of the interface
        """
        inputs = self.generate_interface_inputs(
            structure=self.iface,
            shifts=np.zeros(1, 3),
        )

        total_energy = self.calculate(inputs)[0]

        adhesion_energy = self.get_adhesion_energy(total_energies=total_energy)
        interface_energy = self.get_interface_energy(
            adhesion_energies=adhesion_energy
        )

        return adhesion_energy, interface_energy

    """
        I want SurfaceMatcher(energy_module=IonicPotential) SurfaceEnergy(energy_module=IonicPotential)
        the EnergyModule needs to handle an Interface and Surface object
    """

    def optimizePSO(
        self,
        z_bounds: tp.Optional[tp.List[float]] = None,
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
