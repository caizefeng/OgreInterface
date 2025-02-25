from ase import Atoms
from pymatgen.core import Structure


def structure_to_atoms(structure: Structure) -> Atoms:
    """Manually convert a Pymatgen Structure to an ASE Atoms object."""
    positions = structure.cart_coords
    numbers = structure.atomic_numbers
    cell = structure.lattice.matrix
    return Atoms(positions=positions, numbers=numbers, cell=cell, pbc=True)
