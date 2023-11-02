import typing as tp

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
from ase import Atoms
import numpy as np

from OgreInterface.interfaces.base_interface import BaseInterface
from OgreInterface.lattice_match import OgreMatch
from OgreInterface.surfaces import Surface
from OgreInterface import utils

SelfInterface = tp.TypeVar("SelfInterface", bound="Interface")


class Interface(BaseInterface):
    def __init__(
        self,
        substrate: tp.Union[Surface, SelfInterface],
        film: tp.Union[Surface, SelfInterface],
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
        substrate_strain_fraction: float = 0.0,
    ) -> SelfInterface:
        super().__init__(
            substrate=substrate,
            film=film,
            match=match,
            interfacial_distance=interfacial_distance,
            vacuum=vacuum,
            center=center,
            substrate_strain_fraction=substrate_strain_fraction,
        )

    def get_interface(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the interface structure generated from the OgreMatch

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the interface structure
        """
        if orthogonal:
            if return_atoms:
                return utils.get_atoms(self._orthogonal_structure)
            else:
                return self._orthogonal_structure
        else:
            if return_atoms:
                return utils.get_atoms(self._non_orthogonal_structure)
            else:
                return self._non_orthogonal_structure

    def get_substrate_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the substrate supercell generated from the OgreMatch
        (i.e. the interface structure with the film atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the substrate supercell structure
        """
        if orthogonal:
            if return_atoms:
                return utils.get_atoms(self._orthogonal_substrate_structure)
            else:
                return self._orthogonal_substrate_structure
        else:
            if return_atoms:
                return utils.get_atoms(
                    self._non_orthogonal_substrate_structure
                )
            else:
                return self._non_orthogonal_substrate_structure

    def get_film_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> tp.Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the film supercell generated from the OgreMatch
        (i.e. the interface structure with the substrate atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the film supercell structure
        """
        if orthogonal:
            if return_atoms:
                return utils.get_atoms(self._orthogonal_film_structure)
            else:
                return self._orthogonal_film_structure
        else:
            if return_atoms:
                return utils.get_atoms(self._non_orthogonal_film_structure)
            else:
                return self._non_orthogonal_film_structure

    def replace_species(
        self, site_index: int, species_mapping: tp.Dict[str, str]
    ) -> None:
        """
        This function can be used to replace the species at a given site in the interface structure

        Examples:
            >>> interface.replace_species(site_index=42, species_mapping={"In": "Zn", "As": "Te"})

        Args:
            site_index: Index of the site to be replaced
            species_mapping: Dictionary showing the mapping between species.
                For example if you wanted to replace InAs with ZnTe then the species mapping would
                be as shown in the example above.
        """
        species_str = self._orthogonal_structure[site_index].species_string

        if species_str in species_mapping:
            is_sub = self._non_orthogonal_structure[site_index].properties[
                "is_sub"
            ]
            self._non_orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )
            self._orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )

            if is_sub:
                sub_iface_equiv = np.array(
                    self._orthogonal_substrate_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                sub_site_ind = np.where(sub_iface_equiv == site_index)[0][0]
                self._non_orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
            else:
                film_iface_equiv = np.array(
                    self._orthogonal_film_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                film_site_ind = np.where(film_iface_equiv == site_index)[0][0]
                self._non_orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
        else:
            raise ValueError(
                f"Species: {species_str} is not is species mapping"
            )

    def _load_relaxed_structure(
        self, relaxed_structure_file: str
    ) -> np.ndarray:
        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        desc_str = poscar_str[0].split("|")

        layers = desc_str[0].split("=")[1].split(",")
        termination_index = desc_str[1].split("=")[1].split(",")
        ortho = bool(int(desc_str[2].split("=")[1]))
        d_int = desc_str[3].split("=")[1]
        layers_to_relax = desc_str[4].split("=")[1].split(",")

        film_layers = int(layers[0])
        sub_layers = int(layers[1])

        film_termination_index = int(termination_index[0])
        sub_termination_index = int(termination_index[1])

        N_film_layers_to_relax = int(layers_to_relax[0])
        N_sub_layers_to_relax = int(layers_to_relax[1])

        if (
            d_int == f"{self.interfacial_distance:.3f}"
            and film_termination_index == self.film.termination_index
            and sub_termination_index == self.substrate.termination_index
        ):
            relaxed_structure = Structure.from_file(relaxed_structure_file)

            if ortho:
                unrelaxed_structure = self._orthogonal_structure.copy()
            else:
                unrelaxed_structure = self._non_orthogonal_structure.copy()

            unrelaxed_structure.add_site_property(
                "orig_ind", list(range(len(unrelaxed_structure)))
            )

            unrelaxed_hydrogen_inds = np.where(
                np.array(unrelaxed_structure.atomic_numbers) == 1
            )[0]

            unrelaxed_structure.remove_sites(unrelaxed_hydrogen_inds)

            unrelaxed_structure.add_site_property(
                "orig_ind", list(range(len(unrelaxed_structure)))
            )

            is_negative = np.linalg.det(unrelaxed_structure.lattice.matrix) < 0

            if is_negative:
                relaxed_structure = Structure(
                    lattice=Lattice(relaxed_structure.lattice.matrix * -1),
                    species=relaxed_structure.species,
                    coords=relaxed_structure.frac_coords,
                )

            hydrogen_inds = np.where(
                np.array(relaxed_structure.atomic_numbers) == 1
            )[0]

            relaxed_structure.remove_sites(hydrogen_inds)

            relaxation_shifts = np.zeros((len(unrelaxed_structure), 3))

            is_film_full = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_full = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_full = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )
            sub_to_delete = np.logical_and(
                is_sub_full,
                layer_index_full < self.substrate.layers - sub_layers,
            )

            film_to_delete = np.logical_and(
                is_film_full, layer_index_full >= film_layers
            )

            to_delete = np.where(np.logical_or(sub_to_delete, film_to_delete))[
                0
            ]

            unrelaxed_structure.remove_sites(to_delete)

            is_film_small = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_small = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_small = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )

            film_layers_to_relax = np.arange(N_film_layers_to_relax)

            sub_layers_to_relax = np.arange(
                self.substrate.layers - N_sub_layers_to_relax,
                self.substrate.layers,
            )

            film_to_relax = np.logical_and(
                is_film_small, np.isin(layer_index_small, film_layers_to_relax)
            )
            sub_to_relax = np.logical_and(
                is_sub_small, np.isin(layer_index_small, sub_layers_to_relax)
            )

            relaxed_inds = np.where(
                np.logical_or(film_to_relax, sub_to_relax)
            )[0]

            periodic_shifts = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ).dot(unrelaxed_structure.lattice.matrix)

            ref_ind = np.min(np.where(is_sub_small)[0])
            unrelaxed_ref = unrelaxed_structure[ref_ind].coords
            relaxed_ref = relaxed_structure[ref_ind].coords

            for i in relaxed_inds:
                init_ind = unrelaxed_structure[i].properties["orig_ind"]
                relaxed_coords = relaxed_structure[i].coords
                relaxed_coords -= relaxed_ref
                unrelaxed_coords = unrelaxed_structure[i].coords
                unrelaxed_coords -= unrelaxed_ref

                all_relaxed_coords = periodic_shifts + relaxed_coords
                dists = np.linalg.norm(
                    all_relaxed_coords - unrelaxed_coords, axis=1
                )
                center_ind = np.argmin(dists)
                bond = all_relaxed_coords[center_ind] - unrelaxed_coords
                relaxation_shifts[init_ind] = bond

            return relaxation_shifts
        else:
            raise ValueError(
                "The surface terminations and interfacial distances must be the same"
            )

    def relax_interface(self, relaxed_structure_file: str) -> None:
        """
        This function will shift the positions of the atoms near the interface coorresponding to the
        atomic positions from a relaxed interface. This especially usefull when running DFT on large interface
        structures because the atomic positions can be relaxed using an interface with less layers, and
        then the relax positions can be applied to a much larger interface for a static DFT calculation.

        Examples:
            >>> interface.relax_interface(relax_structure_file="CONTCAR")

        Args:
            relaxed_structure_file: File path to the relax structure (CONTCAR/POSCAR for now)
        """
        relaxation_shifts = self._load_relaxed_structure(
            relaxed_structure_file
        )
        init_ortho_structure = self._orthogonal_structure
        non_hydrogen_inds = np.array(init_ortho_structure.atomic_numbers) != 1
        new_coords = init_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_ortho_structure = Structure(
            lattice=init_ortho_structure.lattice,
            species=init_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_ortho_structure.site_properties,
        )
        (
            relaxed_ortho_film_structure,
            relaxed_ortho_sub_structure,
        ) = self._get_film_and_substrate_parts(relaxed_ortho_structure)

        init_non_ortho_structure = self._non_orthogonal_structure
        non_hydrogen_inds = (
            np.array(init_non_ortho_structure.atomic_numbers) != 1
        )
        new_coords = init_non_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_non_ortho_structure = Structure(
            lattice=init_non_ortho_structure.lattice,
            species=init_non_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_non_ortho_structure.site_properties,
        )
        (
            relaxed_non_ortho_film_structure,
            relaxed_non_ortho_sub_structure,
        ) = self._get_film_and_substrate_parts(relaxed_non_ortho_structure)

        self._orthogonal_structure = relaxed_ortho_structure
        self._orthogonal_film_structure = relaxed_ortho_film_structure
        self._orthogonal_substrate_structure = relaxed_ortho_sub_structure
        self._non_orthogonal_structure = relaxed_non_ortho_structure
        self._non_orthogonal_film_structure = relaxed_non_ortho_film_structure
        self._non_orthogonal_substrate_structure = (
            relaxed_non_ortho_sub_structure
        )

    def write_file(
        self,
        output: str = "POSCAR_interface",
        orthogonal: bool = True,
        relax: bool = False,
        film_layers_to_relax: int = 1,
        substrate_layers_to_relax: int = 1,
        atomic_layers: bool = False,
        relax_z_only: bool = False,
    ) -> None:
        """
        Write the POSCAR of the interface

        Args:
            output: File path of the output POSCAR
            orthogonal: Determines of the orthogonal structure is written
            relax: Determines if selective dynamics is applied to the atoms at the interface
            film_layers_to_relax: Number of unit cell layers near the interface to relax
            substrate_layers_to_relax: Number of unit cell layers near the interface to relax
            atomic_layers: Determines if the number of layer is atomic layers or unit cell layers
            relax_z_only: Determines if the relaxation is in the z-direction only
        """
        if orthogonal:
            slab = self._orthogonal_structure
        else:
            slab = self._non_orthogonal_structure

        comment = "|".join(
            [
                f"L={self.film.layers},{self.substrate.layers}",
                f"T={self.film.termination_index},{self.substrate.termination_index}",
                f"O={int(orthogonal)}",
                f"d={self.interfacial_distance:.3f}",
            ]
        )

        if relax:
            comment += "|" + "|".join(
                [
                    f"R={film_layers_to_relax},{substrate_layers_to_relax}",
                ]
            )
            film_layers = np.arange(film_layers_to_relax)

            if atomic_layers:
                layer_key = "atomic_layer_index"
                sub_layers = np.arange(
                    self.substrate.atomic_layers - substrate_layers_to_relax,
                    self.substrate.atomic_layers,
                )
            else:
                layer_key = "layer_index"
                sub_layers = np.arange(
                    self.substrate.layers - substrate_layers_to_relax,
                    self.substrate.layers,
                )

            layer_index = np.array(slab.site_properties[layer_key])
            is_sub = np.array(slab.site_properties["is_sub"])
            is_film = np.array(slab.site_properties["is_film"])
            film_to_relax = np.logical_and(
                is_film, np.isin(layer_index, film_layers)
            )
            sub_to_relax = np.logical_and(
                is_sub, np.isin(layer_index, sub_layers)
            )

            to_relax = np.repeat(
                np.logical_or(sub_to_relax, film_to_relax).reshape(-1, 1),
                repeats=3,
                axis=1,
            )

            if relax_z_only:
                to_relax[:, :2] = False

        comment += "|" + "|".join(
            [
                f"a={self._a_shift:.4f}",
                f"b={self._b_shift:.4f}",
            ]
        )

        if not self.substrate._passivated and not self.film._passivated:
            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string()

        else:
            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0 and oxi != 0.5:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi == 0.5:
                            H_str = "H.5"
                        elif oxi > 1.0 and oxi != 1.5:
                            H_str = "H" + f"{oxi:.2f}"
                        elif oxi == 1.5:
                            H_str = "H1.5"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)