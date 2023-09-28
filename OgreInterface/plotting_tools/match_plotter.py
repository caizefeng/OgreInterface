import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon, Circle
from OgreInterface.lattice_match import OgreMatch
from pymatgen.core.structure import Structure
import numpy as np


def _sigmoid(
    x: np.ndarray,
    scale: float = 20,
    shift: float = 0.4,
) -> np.ndarray:
    return 1 / (1 + np.exp(scale * (x - shift)))


def _get_cmap(color: str, ascending: bool = True) -> ListedColormap:
    rgb = np.array(to_rgb(color)).reshape(1, -1)
    n_segments = 501
    rgb_list = np.repeat(
        rgb,
        repeats=n_segments,
        axis=0,
    )

    alpha_x = np.linspace(0, 1, n_segments)

    if ascending:
        alpha_y = _sigmoid(x=alpha_x, scale=-20, shift=0.55)
    else:
        alpha_y = _sigmoid(x=alpha_x, scale=20, shift=0.45)

    rgba_list = np.c_[rgb_list, alpha_y]

    cmap = ListedColormap(colors=rgba_list)

    return cmap


def _get_circle_mask(
    bounds: np.ndarray,
    ax: plt.axes,
    radius: float,
    zorder: int = 300,
) -> None:
    grid = np.linspace(bounds[0], bounds[1], 501)
    X, Y = np.meshgrid(grid, grid)
    R = np.sqrt(X**2 + Y**2)
    mask = np.ones(X.shape + (4,))
    mask[R < radius] = np.array([1.0, 1.0, 1.0, 0.0])

    ax.imshow(
        mask,
        extent=np.tile(bounds, 2),
        zorder=zorder,
    )


def _get_gradient(
    color: str,
    ax: plt.axes,
    ascending: bool = True,
    zorder: int = 10,
    extents: list = [0.0, 1.0, 0.0, 1.0],
) -> None:
    rgb = np.array(to_rgb(color))
    gradient = np.ones((501, 501, 4))
    gradient[:, :, :3] *= rgb[None, None, :]

    alpha_x = np.linspace(0, 1, 501)

    if ascending:
        alpha_y = 0.3 * _sigmoid(x=alpha_x, scale=-20, shift=0.5)
    else:
        alpha_y = 0.3 * _sigmoid(x=alpha_x, scale=20, shift=0.5)

    gradient[:, :, -1] *= np.repeat(
        alpha_y.reshape(1, -1), repeats=501, axis=0
    )

    ax.imshow(gradient, zorder=zorder, extent=extents)


def _add_line(
    ax: plt.axes,
    x: np.ndarray,
    y: np.ndarray,
    cmap: ListedColormap,
    vmin: float,
    vmax: float,
    zorder: int = 10,
    linestyle: str = "-",
    linewidth: float = 1.0,
) -> None:
    points = np.c_[x, y].reshape(-1, 1, 2)
    segments = np.concatenate(
        [
            points[:-1],
            points[1:],
        ],
        axis=1,
    )

    norm = Normalize(vmin=vmin, vmax=vmax)

    lc = LineCollection(
        segments=segments,
        cmap=cmap,
        norm=norm,
        zorder=zorder,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    lc.set_array(x)
    ax.add_collection(lc)


def _add_sc_line(
    ax: plt.axes,
    sc_vectors: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_shift: float,
    y_shift: float,
    color: str,
    zorder: int = 10,
    linestyle: str = "-",
    linewidth: float = 0.75,
) -> None:
    inv_sc = np.linalg.inv(sc_vectors)
    cart_points = np.c_[x, y]
    shifted_cart_points = np.c_[x - x_shift, y - y_shift]
    frac_points = np.round(shifted_cart_points.dot(inv_sc), 6)
    is_outside = (frac_points <= 0.0).any(axis=1) | (frac_points >= 1.0).any(
        axis=1
    )
    cart_points[is_outside] = np.nan

    ax.plot(
        cart_points[:, 0],
        cart_points[:, 1],
        color=color,
        zorder=zorder,
        linewidth=linewidth * 1,
    )


def _get_a_to_i(vectors: np.ndarray) -> np.ndarray:
    a_norm = vectors[0] / np.linalg.norm(vectors[0])
    a_to_i = np.array([[a_norm[0], -a_norm[1]], [a_norm[1], a_norm[0]]]).T

    return a_to_i


def _get_strain_matrix(
    sub_vectors: np.ndarray,
    film_vectors: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(film_vectors) @ sub_vectors


def _get_vector_lines(
    y_intercepts: np.ndarray,
    slope: float,
    x_center_shift: float,
    y_center_shift: float,
    shift_vector: np.ndarray,
    n_segments: int,
):
    all_full = False
    counter = 0

    all_xvals = []
    all_yvals = []

    while not all_full:
        signs = [-1, 1]
        is_full = np.zeros(2).astype(bool)

        for i, sign in enumerate(signs):
            if not is_full[i]:
                if np.isnan(slope):
                    x_intercepts = np.zeros(2) + (
                        x_center_shift + (sign * counter * shift_vector[0])
                    )

                    xvals = np.ones(n_segments) * x_intercepts[0]

                    yvals = np.linspace(
                        y_intercepts[0],
                        y_intercepts[1],
                        n_segments,
                    )
                elif np.round(slope, 6) == 0:
                    x_intercepts = np.copy(y_intercepts)

                    xvals = np.linspace(
                        x_intercepts[0],
                        x_intercepts[1],
                        n_segments,
                    )

                    yvals = (
                        slope
                        * (
                            xvals
                            - (
                                x_center_shift
                                + (sign * counter * shift_vector[0])
                            )
                        )
                    ) + (y_center_shift + (sign * counter * shift_vector[1]))
                else:
                    """
                    y = m ( x - (x_center_shift + x_shift)) + (y_center_shift + y_shift)
                    ((y - (y_center_shift + y_shift)) / slope) + (x_center_shift + x_shift)
                    """
                    ints = (
                        (
                            y_intercepts
                            - (
                                (sign * counter * shift_vector[1])
                                + y_center_shift
                            )
                        )
                        / slope
                    ) + (x_center_shift + (sign * counter * shift_vector[0]))

                    x_intercepts = np.clip(
                        ints,
                        y_intercepts[0],
                        y_intercepts[1],
                    )

                    xvals = np.linspace(
                        x_intercepts[0],
                        x_intercepts[1],
                        n_segments,
                    )

                    yvals = (
                        slope
                        * (
                            xvals
                            - (
                                x_center_shift
                                + (sign * counter * shift_vector[0])
                            )
                        )
                    ) + (y_center_shift + (sign * counter * shift_vector[1]))

                all_xvals.append(xvals)
                all_yvals.append(yvals)

                if np.isnan(slope):
                    if np.abs(x_intercepts[0]) > y_intercepts[1]:
                        is_full[i] = True
                else:
                    if (x_intercepts[1] - x_intercepts[0]) == 0 or (
                        np.abs(yvals) > y_intercepts[1]
                    ).all():
                        is_full[i] = True

        if is_full.all():
            all_full = True

        counter += 1

    return all_xvals, all_yvals


def _get_lines(
    vectors: np.ndarray,
    x_center_shift: float,
    y_center_shift: float,
    max_val: float,
) -> np.ndarray:
    n_segments = 501
    half_max = max_val / 2

    a_vec = vectors[0]
    b_vec = vectors[1]

    a_slope, b_slope = np.divide(
        vectors[:, 1],
        vectors[:, 0],
        out=np.nan * np.ones_like(vectors[:, 1]),
        where=vectors[:, 0] != 0,
    )

    bounds = half_max * np.array([-1, 1])

    a_xvals, a_yvals = _get_vector_lines(
        y_intercepts=bounds,
        slope=a_slope,
        x_center_shift=x_center_shift,
        y_center_shift=y_center_shift,
        shift_vector=b_vec,
        n_segments=n_segments,
    )

    b_xvals, b_yvals = _get_vector_lines(
        y_intercepts=bounds,
        slope=b_slope,
        x_center_shift=x_center_shift,
        y_center_shift=y_center_shift,
        shift_vector=a_vec,
        n_segments=n_segments,
    )

    return a_xvals, a_yvals, b_xvals, b_yvals


def _get_sc_origin_coords(
    lattice: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.array:
    lattice_3d = np.eye(3)
    lattice_3d[:2, :2] = lattice
    plot_struc = Structure(
        lattice=lattice_3d,
        species=["H"],
        coords=np.zeros((1, 3)),
        to_unit_cell=True,
        coords_are_cartesian=True,
    )
    plot_struc.make_supercell(transformation_matrix)
    cart_coords = plot_struc.cart_coords[:, :2]

    return cart_coords


def _get_miller_label(miller_index):
    label = []
    for i in miller_index:
        if i < 0:
            label.append("\\overline{" + f"{abs(int(i))}" + "}")
        else:
            label.append(f"{int(i)}")

    return "[" + " ".join(label) + "]"


def _add_sc_labels(
    ax,
    labels,
    vectors,
    fontsize,
    linewidth,
    height,
    x_shift,
    y_shift,
    zorder,
):
    max_vector = np.linalg.norm(vectors, axis=1).max()
    height = 0.05 * max_vector
    xy_shift = np.array([x_shift, y_shift])

    for label, vector in zip(labels, vectors):
        rotation = np.rad2deg(np.arctan2(vector[1], vector[0]))
        if rotation > 0:
            theta = np.deg2rad(rotation)
            rot = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )
            shift = rot.dot(np.array([0, height]))
            va = "bottom"
        else:
            shift = np.array([0, -height])
            va = "top"

        ax.annotate(
            # ((0.5 * vector[0]) + x_shift + (1 * shift[0]),
            # (0.5 * vector[1]) + y_shift + (1 * shift[1]),
            "$" + label + "$",
            xy=(0.5 * vector[:2]) + xy_shift + (1.5 * shift),
            fontsize=fontsize,
            ha="center",
            va=va,
            # bbox=dict(boxstyle="round", fc="w", ec="w", alpha=1),
            zorder=zorder,
            rotation=rotation,
            rotation_mode="anchor",
            transform_rotates_text=True,
        )
        ax.annotate(
            "",
            xytext=vector[:2] + xy_shift + shift,
            xy=xy_shift + shift,
            fontsize=fontsize,
            ha="center",
            va=va,
            arrowprops=dict(
                arrowstyle="<-",
                color="black",
                shrinkA=0,
                shrinkB=0,
                patchA=None,
                patchB=None,
                connectionstyle="arc3,rad=0",
                linewidth=linewidth,
            ),
            zorder=zorder,
        )


def _add_legend(
    ax,
    matrix,
    arrow_color,
    color,
    radius,
    labels,
    fontsize,
    linewidth,
    zorder,
    part="",
):
    norm_matrix = matrix / np.linalg.norm(matrix, axis=1)[:, None]

    for i in range(2):
        t = ax.text(
            radius * norm_matrix[i, 0],
            radius * norm_matrix[i, 1],
            "$" + labels[i] + part + "$",
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", fc="w", ec="black", alpha=1),
            zorder=zorder,
        )
        bb = t.get_bbox_patch()
        ax.annotate(
            "",
            xytext=radius * norm_matrix[i, :2],
            xy=(0, 0),
            fontsize=fontsize,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="<-",
                color=arrow_color,
                shrinkA=5,
                shrinkB=0,
                patchA=bb,
                patchB=None,
                connectionstyle="arc3,rad=0",
                linewidth=linewidth,
            ),
            zorder=zorder,
        )

        circ1 = Circle(
            xy=[0, 0],
            radius=0.03 * radius,
            edgecolor=arrow_color,
            facecolor=arrow_color,
            zorder=zorder + 1,
        )
        ax.add_patch(circ1)


def plot_match(
    match: OgreMatch,
    padding: float = 0.2,
    substrate_color: str = "orange",
    film_color: str = "green",
    output: str = "interface_view.png",
    show_in_colab: bool = False,
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

    a_film_x_vals, a_film_y_vals, b_film_x_vals, b_film_y_vals = _get_lines(
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
    ) = _get_lines(
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
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(6, 4), dpi=400)
    # fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
    for k in axs:
        axs[k].axis("off")
        axs[k].set_aspect("equal")
        axs[k].tick_params(
            left=False, labelleft=False, bottom=False, labelbottom=False
        )
        axs[k].set_xlim(-square_length / 2, square_length / 2)
        axs[k].set_ylim(-square_length / 2, square_length / 2)

    ax = axs["C"]

    lr_cmap = _get_cmap(film_color, ascending=True)
    rl_cmap = _get_cmap(substrate_color, ascending=False)

    _get_gradient(
        color=film_color,
        ax=ax,
        ascending=True,
        zorder=1,
        extents=[-square_length / 2, square_length / 2] * 2,
    )

    _get_gradient(
        color=substrate_color,
        ax=ax,
        ascending=False,
        zorder=2,
        extents=[-square_length / 2, square_length / 2] * 2,
    )

    for x, y in zip(a_film_x_vals, a_film_y_vals):
        _add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=lr_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=10,
        )

        _add_sc_line(
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
        _add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=lr_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=10,
        )

        _add_sc_line(
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
        _add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=rl_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=20,
        )

        _add_sc_line(
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
        _add_line(
            ax=ax,
            x=x,
            y=y,
            cmap=rl_cmap,
            vmin=-square_length / 2,
            vmax=square_length / 2,
            zorder=20,
        )

        _add_sc_line(
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

    sub_sc_a_label = _get_miller_label(match.substrate_sl_basis[0])
    sub_sc_b_label = _get_miller_label(match.substrate_sl_basis[1])
    film_sc_a_label = _get_miller_label(match.film_sl_basis[0])
    film_sc_b_label = _get_miller_label(match.film_sl_basis[1])

    sc_a_label = " ".join(
        [
            f"{int(match.film_sl_scale_factors[0])}",
            film_sc_a_label,
            "_{F}",
            "||",
            f"{int(match.substrate_sl_scale_factors[0])}",
            sub_sc_a_label,
            "_{S}",
        ]
    )

    sc_b_label = " ".join(
        [
            f"{int(match.film_sl_scale_factors[1])}",
            film_sc_b_label,
            "_{F}",
            "||",
            f"{int(match.substrate_sl_scale_factors[1])}",
            sub_sc_b_label,
            "_{S}",
        ]
    )

    _add_sc_labels(
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
    ) = _get_lines(
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
    ) = _get_lines(
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
    _get_circle_mask(
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

    _get_circle_mask(
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

    sub_a_label = _get_miller_label(match.substrate_basis[0])
    sub_b_label = _get_miller_label(match.substrate_basis[1])

    _add_legend(
        ax=axs["A"],
        matrix=aligned_substrate_vectors,
        arrow_color="black",
        color="black",
        radius=radius,
        labels=[sub_a_label, sub_b_label],
        fontsize=12,
        linewidth=1.0,
        zorder=310,
        part="_{S}",
    )

    film_a_label = _get_miller_label(match.film_basis[0])
    film_b_label = _get_miller_label(match.film_basis[1])

    _add_legend(
        ax=axs["B"],
        matrix=aligned_film_vectors,
        arrow_color="black",
        color="black",
        radius=radius,
        labels=[film_a_label, film_b_label],
        fontsize=12,
        linewidth=1.0,
        zorder=310,
        part="_{F}",
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(output, transparent=False)

    if not show_in_colab:
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
