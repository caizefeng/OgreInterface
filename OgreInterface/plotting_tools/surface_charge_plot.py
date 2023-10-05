from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from OgreInterface.generate import SurfaceGenerator

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, Union, List, Dict
import itertools

from OgreInterface import utils


def _get_triangle(
    coords: Tuple[int, int],
    color: Union[str, List[float]],
    is_film: bool,
):
    if is_film:
        xy = np.array(
            [
                [coords[0], coords[1]],
                [coords[0], coords[1] + 1],
                [coords[0] + 1, coords[1] + 1],
                [coords[0], coords[1]],
            ]
        )
    else:
        xy = np.array(
            [
                [coords[0], coords[1]],
                [coords[0] + 1, coords[1]],
                [coords[0] + 1, coords[1] + 1],
                [coords[0], coords[1]],
            ]
        )

    poly = Polygon(
        xy=xy,
        closed=True,
        fc=color,
        ec=(1, 1, 1, 0),
        zorder=1,
    )

    return poly


def _get_square(
    coords: Tuple[int, int],
    color: Union[str, List[float]],
):
    xy = np.array(
        [
            [coords[0], coords[1]],
            [coords[0], coords[1] + 1],
            [coords[0] + 1, coords[1] + 1],
            [coords[0] + 1, coords[1]],
            [coords[0], coords[1]],
        ]
    )

    poly = Polygon(
        xy=xy,
        closed=True,
        fc=color,
        ec=(1, 1, 1, 0),
    )

    return poly


def plot_surface_charge_matrix(
    films: SurfaceGenerator,
    substrates: SurfaceGenerator,
    output: str = "surface_charge_matrix.png",
    dpi: int = 400,
):
    sub_comp = utils.get_latex_formula(substrates[0].formula)
    film_comp = utils.get_latex_formula(films[0].formula)
    sub_miller = utils.get_miller_index_label(substrates[0].miller_index)
    film_miller = utils.get_miller_index_label(films[0].miller_index)

    film_surface_charges = [film.bottom_surface_charge for film in films]
    substrate_surface_charges = [
        substrate.bottom_surface_charge for substrate in substrates
    ]

    x_size = 4
    y_size = 4

    ratio = len(substrate_surface_charges) / len(film_surface_charges)

    if ratio < 1:
        fig_x_size = x_size
        fig_y_size = y_size / ratio
    else:
        fig_x_size = x_size * ratio
        fig_y_size = y_size

    fig, ax = plt.subplots(
        figsize=(fig_x_size, fig_y_size),
        dpi=dpi,
    )

    fontsize = 14

    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(
        f"{sub_comp}({sub_miller}) Termination Index",
        fontsize=fontsize,
    )
    ax.set_ylabel(
        f"{film_comp}({film_miller}) Termination Index",
        fontsize=fontsize,
    )

    cmap_max = max(
        1,
        np.ceil(
            np.abs(
                np.concatenate(
                    [film_surface_charges, substrate_surface_charges]
                )
            ).max()
        ),
    )

    cmap = cm.get_cmap("bwr")
    norm = Normalize(vmin=-cmap_max, vmax=cmap_max)

    color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    inds = itertools.product(range(len(substrates)), range(len(films)))

    bad_inds = []

    for ind in inds:
        film_ind = ind[1]
        sub_ind = ind[0]

        film_charge = film_surface_charges[film_ind]
        sub_charge = substrate_surface_charges[sub_ind]

        film_color = color_mapper.to_rgba(film_charge)
        sub_color = color_mapper.to_rgba(sub_charge)

        film_tri = _get_triangle(
            coords=ind,
            color=film_color,
            is_film=True,
        )
        sub_tri = _get_triangle(
            coords=ind,
            color=sub_color,
            is_film=False,
        )

        ax.add_patch(film_tri)
        ax.add_patch(sub_tri)

        ax.plot(
            [sub_ind, sub_ind + 1],
            [film_ind, film_ind + 1],
            color="black",
            zorder=20,
        )

        if -1.0 < film_charge < 1.0:
            film_sign = 0.0
        elif film_charge <= -1.0:
            film_sign = -1.0
        elif film_charge >= 1.0:
            film_sign = 1.0

        if -1.0 < sub_charge < 1.0:
            sub_sign = 0.0
        elif sub_charge <= -1.0:
            sub_sign = -1.0
        elif sub_charge >= 1.0:
            sub_sign = 1.0

        sign_prod = sub_sign * film_sign

        if sub_sign + film_sign != 0 and sign_prod != 0:
            bad_inds.append(ind)
        #     rect = _get_square(coords=ind, color=(0, 0, 0, 0.6))
        #     ax.add_patch(rect)

    for i in range(len(film_surface_charges)):
        ax.axhline(
            i,
            color="black",
            zorder=20,
        )

    for i in range(len(substrate_surface_charges)):
        ax.axvline(
            i,
            color="black",
            zorder=20,
        )

    ax.set_xlim(0, len(substrate_surface_charges))
    ax.set_ylim(0, len(film_surface_charges))

    ax.set_yticks(
        ticks=np.arange(len(film_surface_charges)) + 0.5,
        labels=[str(i) for i in range(len(film_surface_charges))],
    )

    ax.set_xticks(
        ticks=np.arange(len(substrate_surface_charges)) + 0.5,
        labels=[str(i) for i in range(len(substrate_surface_charges))],
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(
        color_mapper,
        cax=cax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.locator_params(nbins=(2 * cmap_max) + 1)

    cbar.set_label(
        "Residual Surface Charge (Film/Substrate)",
        fontsize=fontsize,
        labelpad=8,
    )

    # r = 0.5  # units
    # # radius in display coordinates:
    # r_ = (
    #     ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
    # )  # points
    # # marker size as the area of a circle
    # marker_size = 2 * r_**2

    # for film_ind, sub_ind in bad_inds:
    #     ax.scatter(
    #         [film_ind + 0.5],
    #         [sub_ind + 0.5],
    #         marker="o",
    #         fc="white",
    #         ec="black",
    #         s=2 * r_,
    #     )
    # print(marker_size)

    ax.tick_params(labelsize=fontsize)

    ax.set_aspect("equal")
    fig.tight_layout(pad=0.4)
    fig.savefig(
        output,
        bbox_inches="tight",
        transparent=False,
    )
