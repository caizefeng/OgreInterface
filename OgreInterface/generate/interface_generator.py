"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from copy import deepcopy
from typing import Union, List, TypeVar, Tuple, Dict, Optional
from itertools import combinations, product, groupby
from collections.abc import Sequence
import math


from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from ase import Atoms
from tqdm import tqdm
import networkx as nx
import numpy as np
import spglib


from OgreInterface import utils
from OgreInterface.lattice_match import ZurMcGill
from OgreInterface.surfaces.base_surface import BaseSurface
from OgreInterface.surfaces.surface import Surface
from OgreInterface.surfaces.molecular_surface import MolecularSurface
from OgreInterface.surfaces.interface import Interface

SelfInterfaceGenerator = TypeVar(
    "SelfInterfaceGenerator", bound="InterfaceGenerator"
)


class TolarenceError(RuntimeError):
    """Class to handle errors when no interfaces are found for a given tolarence setting."""

    pass


class InterfaceGenerator:
    """Class for generating interfaces from two bulk structures

    This class will use the lattice matching algorithm from Zur and McGill to generate
    commensurate interface structures between two inorganic crystalline materials.

    Examples:
        >>> from OgreInterface.generate import SurfaceGenerator, InterfaceGenerator
        >>> subs = SurfaceGenerator.from_file(filename="POSCAR_sub", miller_index=[1,1,1], layers=5)
        >>> films = SurfaceGenerator.from_file(filename="POSCAR_film", miller_index=[1,1,1], layers=5)
        >>> interface_generator = InterfaceGenerator(substrate=subs[0], film=films[0])
        >>> interfaces = interface_generator.generate_interfaces() # List of OgreInterface Interface objects

    Args:
        substrate: Surface class of the substrate material
        film: Surface class of the film materials
        max_area_mismatch: Tolarance of the area mismatch (eq. 2.1 in Zur and McGill)
        max_angle_strain: Tolarence of the angle mismatch between the film and substrate lattice vectors
        max_linear_strain: Tolarence of the length mismatch between the film and substrate lattice vectors
        max_area: Maximum area of the interface unit cell cross section
        interfacial_distance: Distance between the top atom in the substrate to the bottom atom of the film
            If None, the interfacial distance will be predicted based on the average distance of the interlayer
            spacing between the film and substrate materials.
        vacuum: Size of the vacuum in Angstroms
        center: Determines of the interface should be centered in the vacuum

    Attributes:
        substrate (Surface): Surface class of the substrate material
        film (Surface): Surface class of the film materials
        max_area_mismatch (float): Tolarance of the area mismatch (eq. 2.1 in Zur and McGill)
        max_angle_strain (float): Tolarence of the angle mismatch between the film and substrate lattice vectors
        max_linear_strain (float): Tolarence of the length mismatch between the film and substrate lattice vectors
        max_area (float): Maximum area of the interface unit cell cross section
        interfacial_distance (Union[float, None]): Distance between the top atom in the substrate to the bottom atom of the film
            If None, the interfacial distance will be predicted based on the average distance of the interlayer
            spacing between the film and substrate materials.
        vacuum (float): Size of the vacuum in Angstroms
        center: Determines of the interface should be centered in the vacuum
        match_list (List[OgreMatch]): List of OgreMatch objects for each interface generated
    """

    def __init__(
        self,
        substrate: Union[Surface, MolecularSurface, Interface],
        film: Union[Surface, MolecularSurface, Interface],
        max_strain: float = 0.01,
        max_area_mismatch: Optional[float] = None,
        max_area: Optional[float] = None,
        interfacial_distance: Optional[float] = 2.0,
        vacuum: float = 40.0,
        center: bool = False,
        substrate_strain_fraction: float = 0.0,
    ):
        if (
            issubclass(type(substrate), BaseSurface)
            or type(substrate) is Interface
        ):
            self.substrate = substrate
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' or 'ogre.core.Interface' not '{type(substrate).__name__}'"
            )

        if issubclass(type(film), BaseSurface) or type(film) is Interface:
            self.film = film
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' or 'ogre.core.Interface' not '{type(film).__name__}'"
            )

        self.center = center
        self._substrate_strain_fraction = substrate_strain_fraction
        self.max_area_mismatch = max_area_mismatch
        self.max_strain = max_strain
        self.max_area = max_area
        self.interfacial_distance = interfacial_distance
        self.vacuum = vacuum
        self._substrate_point_group_operations = (
            self._get_point_group_operations(
                structure=self.substrate.oriented_bulk._init_bulk,
            )
        )
        self._film_point_group_operations = self._get_point_group_operations(
            structure=self.film.oriented_bulk._init_bulk,
        )
        self.match_list = self._generate_interface_props()

    def _get_point_group_operations(self, structure: Structure) -> np.ndarray:
        sg = SpacegroupAnalyzer(structure)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

    def _generate_interface_props(self):
        zm = ZurMcGill(
            film_vectors=self.film.inplane_vectors,
            substrate_vectors=self.substrate.inplane_vectors,
            film_basis=self.film.crystallographic_basis,
            substrate_basis=self.substrate.crystallographic_basis,
            max_area=self.max_area,
            max_strain=self.max_strain,
            max_area_mismatch=self.max_area_mismatch,
        )
        match_list = zm.run(return_all=True)

        if len(match_list) == 0:
            raise TolarenceError(
                "No interfaces were found, please increase the tolarences."
            )
        elif len(match_list) == 1:
            return match_list
        else:
            film_basis_vectors = []
            sub_basis_vectors = []
            film_scale_factors = []
            sub_scale_factors = []
            for i, match in enumerate(match_list):
                film_basis_vectors.append(match.film_sl_basis)
                sub_basis_vectors.append(match.substrate_sl_basis)
                film_scale_factors.append(match.film_sl_scale_factors)
                sub_scale_factors.append(match.substrate_sl_scale_factors)

            film_basis_vectors = np.round(
                np.vstack(film_basis_vectors)
            ).astype(np.int8)
            sub_basis_vectors = np.round(np.vstack(sub_basis_vectors)).astype(
                np.int8
            )
            film_scale_factors = np.round(
                np.concatenate(film_scale_factors)
            ).astype(np.int8)
            sub_scale_factors = np.round(
                np.concatenate(sub_scale_factors)
            ).astype(np.int8)

            film_map = self._get_miller_index_map(
                self._film_point_group_operations, film_basis_vectors
            )
            sub_map = self._get_miller_index_map(
                self._substrate_point_group_operations, sub_basis_vectors
            )

            split_film_basis_vectors = np.vsplit(
                film_basis_vectors, len(match_list)
            )
            split_sub_basis_vectors = np.vsplit(
                sub_basis_vectors, len(match_list)
            )
            split_film_scale_factors = np.split(
                film_scale_factors, len(match_list)
            )
            split_sub_scale_factors = np.split(
                sub_scale_factors, len(match_list)
            )

            sort_vecs = []

            for i in range(len(split_film_basis_vectors)):
                fb = split_film_basis_vectors[i]
                sb = split_sub_basis_vectors[i]
                fs = split_film_scale_factors[i]
                ss = split_sub_scale_factors[i]
                sort_vec = np.concatenate(
                    [
                        [ss[0]],
                        sub_map[tuple(sb[0])],
                        [ss[1]],
                        sub_map[tuple(sb[1])],
                        [fs[0]],
                        film_map[tuple(fb[0])],
                        [fs[1]],
                        film_map[tuple(fb[1])],
                    ]
                )
                sort_vecs.append(sort_vec)

            sort_vecs = np.vstack(sort_vecs)
            unique_sort_vecs, unique_sort_inds = np.unique(
                sort_vecs, axis=0, return_index=True
            )
            unique_matches = [match_list[i] for i in unique_sort_inds]

            sorted_matches = sorted(
                unique_matches,
                key=lambda x: (
                    round(x.area, 6),
                    round(x.strain, 6),
                    round(x._rotation_distortion, 6),
                ),
            )

            return sorted_matches

    def _get_miller_index_map(self, operations, miller_indices):
        miller_indices = np.unique(miller_indices, axis=0)
        not_used = np.ones(miller_indices.shape[0]).astype(bool)
        op = np.einsum("...ij,jk", operations, miller_indices.T)
        op = op.transpose(2, 0, 1)
        unique_vecs = {}

        for i, vec in enumerate(miller_indices):
            if not_used[i]:
                same_inds = (op == vec).all(axis=2).sum(axis=1) > 0

                if not_used[same_inds].all():
                    same_vecs = miller_indices[same_inds]
                    optimal_vec = self._get_optimal_miller_index(same_vecs)
                    unique_vecs[tuple(optimal_vec)] = list(
                        map(tuple, same_vecs)
                    )
                    not_used[same_inds] = False

        mapping = {}
        for key, value in unique_vecs.items():
            for v in value:
                mapping[v] = key

        return mapping

    def _get_optimal_miller_index(self, vecs):
        diff = np.abs(np.sum(np.sign(vecs), axis=1))
        like_signs = vecs[diff == np.max(diff)]
        if len(like_signs) == 1:
            return like_signs[0]
        else:
            first_max = like_signs[
                np.abs(like_signs)[:, 0] == np.max(np.abs(like_signs)[:, 0])
            ]
            if len(first_max) == 1:
                return first_max[0]
            else:
                second_max = first_max[
                    np.abs(first_max)[:, 1] == np.max(np.abs(first_max)[:, 1])
                ]
                if len(second_max) == 1:
                    return second_max[0]
                else:
                    return second_max[
                        np.argmax(np.sign(second_max).sum(axis=1))
                    ]

    def _build_interface(self, match):
        if self.interfacial_distance is None:
            # TODO: Move this to the interface generator
            # Get the distance from the next atomic layer if you were to extend
            # from the top of the substrate
            sub_obs = self.substrate.oriented_bulk.oriented_bulk_structure
            sub_c = sub_obs.frac_coords[:, -1]
            sub_h = self.substrate.oriented_bulk.layer_thickness
            top_layer_dist = np.abs((sub_c.min() + 1) - sub_c.max()) * sub_h

            # Get the distance from the next atomic layer if you were to extend
            # the structure from the bottom
            film_obs = self.film.oriented_bulk.oriented_bulk_structure
            film_c = film_obs.frac_coords[:, -1]
            film_h = self.film.oriented_bulk.layer_thickness
            bottom_layer_dist = (
                np.abs(film_c.min() - (film_c.max() - 1)) * film_h
            )
            i_dist = (top_layer_dist + bottom_layer_dist) / 2
        else:
            i_dist = self.interfacial_distance

        interface = Interface(
            substrate=self.substrate,
            film=self.film,
            interfacial_distance=i_dist,
            match=match,
            vacuum=self.vacuum,
            center=self.center,
            substrate_strain_fraction=self._substrate_strain_fraction,
        )
        return interface

    def generate_interfaces(self):
        """Generates a list of Interface objects from that matches found using the Zur and McGill lattice matching algorithm"""
        interfaces = []

        print(
            f"Generating Interfaces for {self.film.formula_with_miller}[{self.film.termination_index}] and {self.substrate.formula_with_miller}[{self.substrate.termination_index}]:"
        )
        for match in tqdm(self.match_list, dynamic_ncols=True):
            interface = self._build_interface(match=match)
            interfaces.append(interface)

        return interfaces
