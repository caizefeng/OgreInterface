import typing as tp

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Circle
import numpy as np

from OgreInterface.lattice_match import OgreMatch
from OgreInterface.plotting_tools import plotting_utils


def plot_match(
    match: OgreMatch,
    padding: float = 0.2,
    substrate_color: str = "green",
    film_color: str = "orange",
    substrate_label: str = "A",
    film_label: str = "B",
    output: str = "interface_view.png",
    display_results: bool = False,
    dpi: int = 400,
    film_composition: tp.Optional[str] = None,
    substrate_composition: tp.Optional[str] = None,
):
    strain_matrix = match.film_to_substrate_strain_transform[:2, :2]
    film_align_matrix = match.film_align_transform[:2, :2]
    substrate_align_matrix = match.substrate_align_transform[:2, :2]

    substrate_sl_vectors = match.substrate_sl_vectors[:, :2]
    substrate_vectors = match.substrate_vectors[:, :2]
    film_vectors = match.film_vectors[:, :2]

    aligned_film_vectors = film_vectors @ film_align_matrix
    aligned_film_vectors = aligned_film_vectors @ strain_matrix

    aligned_substrate_sl_vectors = (
        substrate_sl_vectors @ substrate_align_matrix
    )
    aligned_substrate_vectors = substrate_vectors @ substrate_align_matrix

    aligned_substrate_sl_vectors = np.round(aligned_substrate_sl_vectors, 6)
    aligned_substrate_vectors = np.round(aligned_substrate_vectors, 6)
    aligned_film_vectors = np.round(aligned_film_vectors, 6)

    min_xy = ((-1 * padding) * np.ones(2)).dot(aligned_substrate_sl_vectors)
    max_xy = ((1 + padding) * np.ones(2)).dot(aligned_substrate_sl_vectors)
    shift = (0.5 * np.ones(2)).dot(aligned_substrate_sl_vectors)

    x_shift = -shift[0]
    y_shift = -shift[1]

    square_length = (max_xy - min_xy).max()
    square_length = np.abs(max_xy - min_xy).max()

    (
        a_film_x_vals,
        a_film_y_vals,
        b_film_x_vals,
        b_film_y_vals,
    ) = plotting_utils._get_lines(
        vectors=aligned_film_vectors,
        x_center_shift=x_shift,
        y_center_shift=y_shift,
        max_val=square_length,
    )

    (
        a_substrate_x_vals,
        a_substrate_y_vals,
        b_substrate_x_vals,
        b_substrate_y_vals,
    ) = plotting_utils._get_lines(
        vectors=aligned_substrate_vectors,
        x_center_shift=x_shift,
        y_center_shift=y_shift,
        max_val=square_length,
    )

    # TODO: Add non gradient polygons to the center sl cell

    mosaic = """
        ACC
        BCC
    """
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(6, 4), dpi=dpi)
    for k in axs:
        axs[k].axis("off")
        axs[k].set_aspect("equal")
        axs[k].tick_params(
            left=False, labelleft=False, bottom=False, labelbottom=False
        )
        axs[k].set_xlim(-square_length / 2, square_length / 2)
        axs[k].set_ylim(-square_length / 2, square_length / 2)

    ax = axs["C"]

    lr_cmap = plotting_utils._get_cmap(film_color, ascending=True)
    rl_cmap = plotting_utils._get_cmap(substrate_color, ascending=False)

    plotting_utils._get_gradient(
        color=film_color,
        ax=ax,
        ascending=True,
        zorder=1,
        extents=[-square_length / 2, square_length / 2] * 2,
    )

    plotting_utils._get_gradient(
        color=substrate_color,
        ax=ax,
        ascending=False,
        zorder=2,
        extents=[-square_length / 2, square_length / 2] * 2,
    )

    for x, y in zip(a_film_x_vals, a_film_y_vals):
        plotting_utils._add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=lr_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=10,
        )

        plotting_utils._add_sc_line(
            ax=ax,
            sc_vectors=aligned_substrate_sl_vectors,
            x=x,
            y=y,
            x_shift=x_shift,
            y_shift=y_shift,
            zorder=10,
            color=film_color,
        )

    for x, y in zip(b_film_x_vals, b_film_y_vals):
        plotting_utils._add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=lr_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=10,
        )

        plotting_utils._add_sc_line(
            ax=ax,
            sc_vectors=aligned_substrate_sl_vectors,
            x=x,
            y=y,
            x_shift=x_shift,
            y_shift=y_shift,
            zorder=10,
            color=film_color,
        )

    for x, y in zip(a_substrate_x_vals, a_substrate_y_vals):
        plotting_utils._add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=rl_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=20,
        )

        plotting_utils._add_sc_line(
            ax=ax,
            sc_vectors=aligned_substrate_sl_vectors,
            x=x,
            y=y,
            x_shift=x_shift,
            y_shift=y_shift,
            zorder=10,
            color=substrate_color,
        )

    for x, y in zip(b_substrate_x_vals, b_substrate_y_vals):
        plotting_utils._add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=rl_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=20,
        )

        plotting_utils._add_sc_line(
            ax=ax,
            sc_vectors=aligned_substrate_sl_vectors,
            x=x,
            y=y,
            x_shift=x_shift,
            y_shift=y_shift,
            zorder=10,
            color=substrate_color,
        )

    poly_color = (np.array([0, 0, 0]) / 255).tolist()

    ax.plot(
        [
            -square_length / 2,
            -square_length / 2,
            square_length / 2,
            square_length / 2,
            -square_length / 2,
        ],
        [
            -square_length / 2,
            square_length / 2,
            square_length / 2,
            -square_length / 2,
            -square_length / 2,
        ],
        color="black",
        linewidth=1,
        zorder=100,
    )

    poly_frac_xy = (
        np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0],
            ]
        )
        - 0.5
    )
    poly = Polygon(
        xy=poly_frac_xy.dot(aligned_substrate_sl_vectors),
        closed=True,
        ec="black",
        fc=poly_color + [0.3],
        zorder=30,
    )
    ax.add_patch(poly)

    sub_sc_a_label = plotting_utils._get_miller_label(
        match.substrate_sl_basis[0]
    )
    sub_sc_b_label = plotting_utils._get_miller_label(
        match.substrate_sl_basis[1]
    )
    film_sc_a_label = plotting_utils._get_miller_label(match.film_sl_basis[0])
    film_sc_b_label = plotting_utils._get_miller_label(match.film_sl_basis[1])

    sc_a_label = " ".join(
        [
            f"{int(match.film_sl_scale_factors[0])}",
            film_sc_a_label,
            "_{" + film_label + "}",
            "||",
            f"{int(match.substrate_sl_scale_factors[0])}",
            sub_sc_a_label,
            "_{" + substrate_label + "}",
        ]
    )

    sc_b_label = " ".join(
        [
            f"{int(match.film_sl_scale_factors[1])}",
            film_sc_b_label,
            "_{" + film_label + "}",
            "||",
            f"{int(match.substrate_sl_scale_factors[1])}",
            sub_sc_b_label,
            "_{" + substrate_label + "}",
        ]
    )

    plotting_utils._add_sc_labels(
        ax=axs["C"],
        labels=[sc_a_label, sc_b_label],
        vectors=aligned_substrate_sl_vectors,
        fontsize=12,
        linewidth=1.0,
        height=1.0,
        x_shift=x_shift,
        y_shift=y_shift,
        zorder=205,
    )

    (
        a_film_leg_x_vals,
        a_film_leg_y_vals,
        b_film_leg_x_vals,
        b_film_leg_y_vals,
    ) = plotting_utils._get_lines(
        vectors=aligned_film_vectors,
        x_center_shift=0.0,
        y_center_shift=0.0,
        max_val=square_length,
    )

    (
        a_substrate_leg_x_vals,
        a_substrate_leg_y_vals,
        b_substrate_leg_x_vals,
        b_substrate_leg_y_vals,
    ) = plotting_utils._get_lines(
        vectors=aligned_substrate_vectors,
        x_center_shift=0.0,
        y_center_shift=0.0,
        max_val=square_length,
    )

    for x, y in zip(a_film_leg_x_vals, a_film_leg_y_vals):
        axs["B"].plot(
            x,
            y,
            color=film_color,
            linewidth=0.75,
        )

    for x, y in zip(b_film_leg_x_vals, b_film_leg_y_vals):
        axs["B"].plot(
            x,
            y,
            color=film_color,
            linewidth=0.75,
        )

    for x, y in zip(a_substrate_leg_x_vals, a_substrate_leg_y_vals):
        axs["A"].plot(
            x,
            y,
            color=substrate_color,
            linewidth=0.75,
        )

    for x, y in zip(b_substrate_leg_x_vals, b_substrate_leg_y_vals):
        axs["A"].plot(
            x,
            y,
            color=substrate_color,
            linewidth=0.75,
        )

    radius = 0.8 * (square_length / 2)
    plotting_utils._get_circle_mask(
        bounds=np.array([-square_length / 2, square_length / 2]),
        ax=axs["A"],
        radius=radius,
    )

    sub_circ = Circle(
        xy=[0, 0],
        radius=radius,
        ec="black",
        fc=to_rgb(substrate_color) + (0.1,),
        linewidth=1,
        zorder=301,
    )
    axs["A"].add_patch(sub_circ)

    plotting_utils._get_circle_mask(
        bounds=np.array([-square_length / 2, square_length / 2]),
        ax=axs["B"],
        radius=radius,
    )

    film_circ = Circle(
        xy=[0, 0],
        radius=radius,
        ec="black",
        fc=to_rgb(film_color) + (0.1,),
        linewidth=1,
        zorder=301,
    )
    axs["B"].add_patch(film_circ)

    sub_a_label = plotting_utils._get_miller_label(match.substrate_basis[0])
    sub_b_label = plotting_utils._get_miller_label(match.substrate_basis[1])

    plotting_utils._add_legend(
        ax=axs["A"],
        matrix=aligned_substrate_vectors,
        arrow_color="black",
        color="black",
        radius=radius,
        labels=[sub_a_label, sub_b_label],
        fontsize=12,
        linewidth=1.0,
        zorder=310,
        part="_{" + substrate_label + "}",
    )

    film_a_label = plotting_utils._get_miller_label(match.film_basis[0])
    film_b_label = plotting_utils._get_miller_label(match.film_basis[1])

    plotting_utils._add_legend(
        ax=axs["B"],
        matrix=aligned_film_vectors,
        arrow_color="black",
        color="black",
        radius=radius,
        labels=[film_a_label, film_b_label],
        fontsize=12,
        linewidth=1.0,
        zorder=310,
        part="_{" + film_label + "}",
    )

    legend_pad = 0.03

    if substrate_composition is not None:
        ax.annotate(
            substrate_composition,
            xy=(legend_pad, 1 - legend_pad),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round",
                fc=[(0.05 * c) + 0.95 for c in to_rgb(substrate_color)],
                ec=[(0.3 * c) + 0.7 for c in to_rgb(substrate_color)],
                alpha=1.0,
            ),
            fontsize=12,
            zorder=1000,
            color=substrate_color,
            weight="bold",
        )

    if film_composition is not None:
        ax.annotate(
            film_composition,
            xy=(1 - legend_pad, 1 - legend_pad),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round",
                fc=[(0.05 * c) + 0.95 for c in to_rgb(film_color)],
                ec=[(0.3 * c) + 0.7 for c in to_rgb(film_color)],
                alpha=1.0,
            ),
            fontsize=12,
            zorder=1000,
            color=film_color,
            weight="bold",
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(output, transparent=False)

    if not display_results:
        plt.close(fig)


if __name__ == "__main__":
    from OgreInterface.generate import SurfaceGenerator, InterfaceGenerator

    subs = SurfaceGenerator.from_file(
        "./poscars/POSCAR_InAs_conv",
        miller_index=[1, 1, 0],
    )

    films = SurfaceGenerator.from_file(
        "./poscars/POSCAR_Al_conv",
        miller_index=[1, 1, 0],
    )

    iface_gen = InterfaceGenerator(
        substrate=subs[0],
        film=films[0],
        # max_area=300,
        # max_area=200,
        # max_area=200.0,
    )

    ifaces = iface_gen.generate_interfaces()
    iface = ifaces[1]
    # iface.write_file("POSCAR_iface_old")

    # iface.plot_interface()
    print(iface)

    match = iface.match

    plot_match(
        match=match,
        padding=0.2,
        substrate_color="orange",
        film_color="green",
    )
