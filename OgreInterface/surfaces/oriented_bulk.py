"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
import copy
import typing as tp
import itertools
from collections.abc import Sequence

from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.operations import SymmOp
import numpy as np
import spglib

from OgreInterface import utils

SelfOrientedBulk = tp.TypeVar("SelfOrientedBulk", bound="OrientedBulk")


class OrientedBulk(Sequence):
    def __init__(
        self,
        bulk: Structure,
        miller_index: tp.List[int],
        make_planar: bool = True,
    ) -> SelfOrientedBulk:
        # Determine if the structure has a hexagonal lattice
        self._is_hexagonal = bulk.lattice.is_hexagonal

        # If the structure is hexagonal and the miller index
        # list has 4 elements then convert it back to cubic
        # notation. (i.e. (hkl) -> (hkil))
        if self._is_hexagonal and len(miller_index) == 4:
            self._init_miller_index = utils.hex_to_cubic_plane(
                hkil=miller_index
            )
        else:
            self._init_miller_index = np.array(miller_index)

        # Determines is the OBS should be oriented to the inplane vectors
        # are in the xy cartesian plane
        self._make_planar = make_planar

        # Input bulk structure
        self._init_bulk = bulk

        # Primitive bulk structure
        self._prim_bulk = self._get_primitive_bulk_structure(bulk)

        # Get the symmetry dataset from spglib
        self._symmetry_dataset = self._get_symmetry_dataset()

        # Add symmetry info to the initial bulk structure
        self._add_symmetry_info(structure=self._init_bulk, is_primitive=False)

        # Add symmetry info to the primitive bulk structure
        self._add_symmetry_info(structure=self._prim_bulk, is_primitive=True)

        # self._init_miller_index = np.array(miller_index)

        # Surface normal, and normalized surface normal
        (
            self._surface_normal,
            self._unit_surface_normal,
        ) = self._get_normal_vector()

        # Primitive bulk structure and primitive equivalent miller index
        (
            self.bulk,
            self.miller_index,
        ) = self._get_bulk_structure_and_miller_index()

        # Transformation matrix to transform self.bulk to self.obs
        self._transformation_matrix = self._get_transformation_matrix()

        # OBS and crystallographic basis
        (
            self._oriented_bulk_structure,
            self._crystallographic_basis,
        ) = self._get_oriented_bulk_structure()

    def __getitem__(self, i) -> PeriodicSite:
        return self._oriented_bulk_structure[i]

    def __len__(self) -> int:
        return len(self._oriented_bulk_structure)

    def __str__(self) -> str:
        return self._oriented_bulk_structure.__str__()

    @property
    def oriented_bulk_structure(self) -> Structure:
        """
        This will return the oriented bulk structure oriented so the a-b
        lattice vectors are in the xy-plane with the a-vector pointing
        along the [1, 0, 0] cartesian direction.
        """
        return self._oriented_bulk_structure

    @property
    def crystallographic_basis(self) -> np.ndarray:
        """
        This will return the crystallographic basis of the oriented bulk
        structure in the reference frame of the input structure.
        """
        return self._crystallographic_basis

    @property
    def layer_thickness(self) -> float:
        """
        This will return the z-height of the oriented bulk structure
        which is effectively the thickness of a unit layer of the slab
        """
        return self._get_projection_along_norm()

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        This returns the transformation matrix used to transform the basis
        of the primitive cell of the input structure to the oriented bulk
        structure using the create_supercell method on a pymatgen structure.
        This transformation matrix can be used for band unfolding.
        """
        return self._transformation_matrix

    @property
    def surface_normal(self) -> np.ndarray:
        """
        This returns the surface normal for the oriented bulk structure if
        make_planar=False this should be the same as self._unit_surface_normal
        if make_planar=True this should be [0, 0, 1].
        """
        a, b, _ = self._oriented_bulk_structure.lattice.matrix
        normal_vec = np.cross(a, b)
        normal_vec /= np.linalg.norm(normal_vec)

        return normal_vec

    @property
    def site_properties(self) -> tp.Dict[str, Sequence]:
        return self._oriented_bulk_structure.site_properties

    @property
    def area(self) -> float:
        """
        Cross section area of the slab in Angstroms^2

        Examples:
            >>> surface.area
            >>> 62.51234

        Returns:
            Cross-section area in Angstroms^2
        """
        inplane_vectors = self.inplane_vectors
        area = np.linalg.norm(np.cross(inplane_vectors[0], inplane_vectors[1]))

        return area

    @property
    def inplane_vectors(self) -> np.ndarray:
        """
        In-plane cartesian vectors of the slab structure

        Examples:
            >>> surface.inplane_vectors
            >>> [[4.0 0.0 0.0]
            ...  [2.0 2.0 0.0]]

        Returns:
            (2, 3) numpy array containing the cartesian coordinates of the in-place lattice vectors
        """
        matrix = copy.deepcopy(self._oriented_bulk_structure.lattice.matrix)

        return matrix[:2]

    def add_charges(self) -> None:
        obs = self.oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_guess()
        charges = [s._oxi_state for s in obs.species]
        self.add_site_property("charge", charges)

    def translate_sites(
        self,
        vector: np.ndarray,
        frac_coords: bool = True,
    ) -> None:
        self._oriented_bulk_structure.translate_sites(
            indices=range(len(self._oriented_bulk_structure)),
            vector=vector,
            frac_coords=frac_coords,
            to_unit_cell=True,
        )

    def round(self, tol: int = 6) -> None:
        self._oriented_bulk_structure = utils.get_rounded_structure(
            structure=self._oriented_bulk_structure,
            tol=tol,
        )

    def add_site_property(self, property_name: str, value: Sequence) -> None:
        self._oriented_bulk_structure.add_site_property(property_name, value)

    def _get_projection_along_norm(self) -> float:
        return np.dot(
            self._oriented_bulk_structure.lattice.matrix[-1],
            self.surface_normal,
        )

    def _get_symmetry_dataset(
        self,
    ) -> tp.Dict[str, np.ndarray]:
        lattice = self._init_bulk.lattice.matrix
        positions = self._init_bulk.frac_coords
        numbers = np.array(self._init_bulk.atomic_numbers)
        cell = (lattice, positions, numbers)

        dataset = spglib.get_symmetry_dataset(cell)

        return dataset

    def _add_symmetry_info(
        self,
        structure: Structure,
        is_primitive: bool = False,
    ) -> None:
        dataset = self._symmetry_dataset
        if is_primitive:
            prim_mapping = dataset["mapping_to_primitive"]
            _, prim_inds = np.unique(prim_mapping, return_index=True)

            structure.add_site_property(
                "bulk_wyckoff",
                [dataset["wyckoffs"][i] for i in prim_inds],
            )
            structure.add_site_property(
                "bulk_equivalent",
                dataset["equivalent_atoms"][prim_inds].tolist(),
            )
        else:
            structure.add_site_property(
                "bulk_wyckoff",
                dataset["wyckoffs"],
            )

            structure.add_site_property(
                "bulk_equivalent",
                dataset["equivalent_atoms"].tolist(),
            )

    def _get_primitive_bulk_structure(self, structure: Structure) -> Structure:
        primitive_bulk = utils.spglib_standardize(
            structure=self._init_bulk,
            to_primitive=True,
            no_idealize=True,
        )

        return primitive_bulk

    def _get_bulk_structure_and_miller_index(
        self,
    ) -> tp.Tuple[Structure, tp.List[int]]:
        bulk = self._init_bulk
        prim_bulk = self._prim_bulk
        miller_index = self._init_miller_index

        if len(bulk) != len(prim_bulk):
            # Primitive bulk lattice (these could be the same)
            lattice = prim_bulk.lattice

            # Get the normal vector of the surface using the metric tensor
            # The metric tensor transforms from fractional recoprocal coords
            # to fractional real space coords
            normal_vector = self._surface_normal

            # Get the surface normal in the basis of the primitive lattice
            prim_normal_vector = lattice.get_fractional_coords(normal_vector)

            # Get convert this to fractional reciprocal coords using the metric
            # tensor to get the primitive equivalent (hkl)
            eq_miller_index = prim_normal_vector.dot(lattice.metric_tensor)

            # Get the reduced miller index (i.e. (2, 2, 2) --> (1, 1, 1))
            eq_miller_index = utils._get_reduced_vector(eq_miller_index)

            return prim_bulk, eq_miller_index.astype(int)
        else:
            return bulk, miller_index

    def _get_normal_vector(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        # Bulk lattice
        lattice = self._init_bulk.lattice

        # Bulk reciprocal lattice
        recip_metric_tensor = (
            lattice.reciprocal_lattice_crystallographic.metric_tensor
        )

        # Miller index of the surface
        miller_index = self._init_miller_index

        # Get the normal vector of the lattice plane using the metric tensor
        # The metric tensor transforms from fractional recoprocal coords to
        # fractional real space coords
        normal_vector = lattice.get_cartesian_coords(
            miller_index.dot(recip_metric_tensor)
        )

        # Normalie the normal vector to length = 1.0
        unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

        return normal_vector, unit_normal_vector

    def _get_inplane_vectors(self) -> tp.List[float]:
        # Get the number of zero elements in (h, k ,l)
        zeros_mask = self.miller_index == 0.0

        # Get the fractional intercepts along the (a, b, c) vectors
        fractional_intercepts = np.array(
            [1.0 / i if i != 0.0 else 0.0 for i in self.miller_index]
        )

        # Expand the fractional intercepts along the (a, b, c) vectors
        # into a (3, 3) with the intercepts along the diagonal
        intercept_matrix = np.eye(3) * fractional_intercepts

        # When any (h, k, l) then its intercept is shifted by its corresponding
        # lattice vector (a, b, c). (i.e. if h == 0) one intercept point gets
        # shifted by a)
        shifts = np.eye(3)
        shifts[~zeros_mask] *= 0.0

        # Add the shifts to get the total intercept matrix
        shifted_intercept_matrix = intercept_matrix + shifts

        # Create a list of know intercept points (i.e. non-zero intercept)
        # one of these will be used as the center point to calculate the
        # inplane vectors later in this function
        known_intercept_points = []
        for i, intercept in enumerate(fractional_intercepts):
            if intercept != 0.0:
                # If the intercept is non-zero then append the point to the
                # known_intercept_points list along with the (0, 1, 2) index
                point = np.zeros(3)
                point[i] = intercept
                known_intercept_points.append((i, point))

        if zeros_mask.sum() > 0:
            # If there is more than one zero then use one of the known
            # intercept points as the center point to calculate the
            # inplane basis vectors of the surface

            # Pick of the known intercept points to use as the center point
            # If there is more than one (i.e. (h, k, 0)) then it doesn't matter
            # because the basis will be reduced using the Zur and McGill algo
            i, known_intercept_point = known_intercept_points[0]

            # Translate intercepts by the center point
            possible_intercepts = np.copy(shifted_intercept_matrix)
            possible_intercepts[zeros_mask] += known_intercept_point

            # The edge points (i.e. ends of the inplane vectors)
            # are the other two intercept points
            edge_points = possible_intercepts[np.arange(3) != i]

            # Calculate the fractional basis vectors
            frac_vec0 = utils._get_reduced_vector(
                edge_points[0] - known_intercept_point
            )
            frac_vec1 = utils._get_reduced_vector(
                edge_points[1] - known_intercept_point
            )
        else:
            # If all elements are non-zero then Calculate the fractional
            # basis vectors by selecting one of the intercept points as the
            # center and the other two as the vector ends
            frac_vec0 = utils._get_reduced_vector(
                shifted_intercept_matrix[0] - shifted_intercept_matrix[1]
            )
            frac_vec1 = utils._get_reduced_vector(
                shifted_intercept_matrix[2] - shifted_intercept_matrix[1]
            )

        # Get the cartesian vectors for lattice vector reduction
        cart_vec0 = self.bulk.lattice.get_cartesian_coords(frac_vec0)
        cart_vec1 = self.bulk.lattice.get_cartesian_coords(frac_vec1)

        # Use the zur and mcgill lattice vector reduction algorithm to
        # get the reduced surface vectors in a right-handed basis
        (
            reduced_cart_vec0,
            reduced_cart_vec1,
            reduced_mat,
        ) = utils.reduce_vectors_zur_and_mcgill(
            cart_vec0,
            cart_vec1,
            surface_normal=self._unit_surface_normal,
        )

        # Get the initial basis matrix before reduction
        init_basis = np.eye(3)
        init_basis[0] = frac_vec0
        init_basis[1] = frac_vec1

        # Get the reduced basis after apply the zur and mcgill reduction algo
        reduced_basis = utils.get_reduced_basis(reduced_mat @ init_basis)

        # Extract the a- and b-vectors (c-vector is not important here)
        a_vec, b_vec, _ = reduced_basis

        return a_vec, b_vec

    def _get_out_of_plane_vector(
        self,
        a_vector: np.ndarray,
        b_vector: np.ndarray,
    ) -> np.ndarray:
        # Lattice of the bulk structure
        lattice = self.bulk.lattice

        # Inter-layer distance of the lattice plane
        d_hkl = lattice.d_hkl(self.miller_index)

        # Stack the a- and b-vectors into a (2x3) matrix
        ab_vecs = np.vstack([a_vector, b_vector])

        # Max value of the out of plane vector elements (u, v, w)
        max_normal = 2
        index_range = sorted(
            reversed(range(-max_normal, max_normal + 1)),
            key=lambda x: abs(x),
        )
        candidates = []
        for c_vec in itertools.product(index_range, index_range, index_range):
            # Loop through all possible cominations of (u, v, w)'s
            possible_basis = np.vstack([ab_vecs, c_vec])
            det = np.linalg.det(lattice.get_cartesian_coords(possible_basis))
            if (not any(c_vec)) or det < 1e-8:
                continue

            # Get the cartesian vector
            cart_c_vec = lattice.get_cartesian_coords(c_vec)

            # Get the projected length of the c vector along the surface normal
            proj = np.dot(cart_c_vec, self._unit_surface_normal)

            # Get the difference between the projected length of the c-vector
            # and the interplanar distance
            diff = np.abs(proj - d_hkl)

            # Calculate the cosine similarity between the surface normal and c
            vec_length = np.linalg.norm(cart_c_vec)
            cosine = np.dot(cart_c_vec / vec_length, self._unit_surface_normal)

            # Add elements to the candidates list
            candidates.append(
                (
                    c_vec,
                    np.round(cosine, 5),
                    np.round(vec_length, 5),
                    np.round(diff, 5),
                )
            )

            if abs(abs(cosine) - 1) < 1e-8 and diff < 1e-8:
                # If cosine of 1 is found, no need to search further.
                break

        # We want the indices with the maximum absolute cosine,
        # but smallest possible length.
        opt_c_vec, _, _, _ = max(
            candidates,
            key=lambda x: (-x[3], x[1], -x[2]),
        )

        return np.array(opt_c_vec)

    def _get_transformation_matrix(self) -> np.ndarray:
        a_vector, b_vector = self._get_inplane_vectors()
        c_vector = self._get_out_of_plane_vector(
            a_vector=a_vector,
            b_vector=b_vector,
        )

        return np.vstack([a_vector, b_vector, c_vector])

    def _get_oriented_bulk_structure(self) -> tp.Tuple[Structure, np.ndarray]:
        # Get the transformation matrix to transform the bulk to the OBS
        transformation_matrix = self._transformation_matrix

        # Make the supercell and then round and mod the coordinates of the new
        # structure. Rounding & modding addresses the case when a fractional
        # coordinate is 0.9999999 but should be modded to 0.0
        _obs = self.bulk.copy().make_supercell(transformation_matrix)
        obs = utils.get_rounded_structure(structure=_obs, tol=6)

        # Add the obs bulk equivalent property
        obs.add_site_property(
            "oriented_bulk_equivalent",
            list(range(len(obs))),
        )

        # Get the cartesian basis vectors of the OBS
        matrix = obs.lattice.matrix

        # Convert the cartesian basis vectors to fractional coordinates in
        # the reference frame of the initial bulk structure
        conv_basis = self._init_bulk.lattice.get_fractional_coords(
            cart_coords=matrix
        )

        # Reduce the basis to integers to get the crystallographic reference
        # frame of the OBS structure
        crystallographic_basis = utils.get_reduced_basis(
            np.round(conv_basis, 6)
        )

        if self._make_planar:
            # Calculate the rotational transformation to orient the OBS so the
            # a and b lattice vectors are in the cartesian xy-plane with a-vec
            # pointing along the [1, 0, 0] direction.
            a_norm = matrix[0] / np.linalg.norm(matrix[0])
            cross_ab = np.cross(matrix[0], matrix[1])
            cross_ab /= np.linalg.norm(cross_ab)
            cross_ca = np.cross(cross_ab, matrix[0])
            cross_ca /= np.linalg.norm(cross_ca)

            ortho_basis = np.vstack([a_norm, cross_ca, cross_ab])

            to_planar_operation = SymmOp.from_rotation_and_translation(
                ortho_basis,
                translation_vec=np.zeros(3),
            )

            # Apply the rotation operation
            obs.apply_operation(to_planar_operation)

        obs = utils.get_rounded_structure(
            structure=obs,
            tol=6,
        )

        return obs, crystallographic_basis


if __name__ == "__main__":
    bulk = Structure.from_file(
        "../../../ogre-stuff/ita/workflow_tests/cifs/InAs.cif"
        # "../../../ogre-stuff/ita/workflow_tests/cifs/Pb3S2Cl2_cubic.cif"
    )
    obs_gen = OrientedBulk(
        bulk=bulk,
        miller_index=[1, 1, 0],
        make_planar=False,
    )

    obs_gen2 = copy.deepcopy(obs_gen)
    obs_gen2._oriented_bulk_structure = "TEST"

    print(obs_gen._oriented_bulk_structure)
    print(obs_gen2._oriented_bulk_structure)
    # print(np.round(obs_gen.oriented_bulk_structure.lattice.matrix, 6))
    # print(obs_gen.layer_thickness)
    # obs = obs_gen.oriented_bulk_structure
    # utils.calculate_possible_shifts(structure=obs, tol=None)
    # print(obs_gen._unit_surface_normal)
    # print(obs_gen.surface_normal)
    # print(obs.miller_index)
    # print(obs._init_miller_index)
    # obs._get_oriented_bulk_structure()
    # obs._get_crystallographic_basis()
