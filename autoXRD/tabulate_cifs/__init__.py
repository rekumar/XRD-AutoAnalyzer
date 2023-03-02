from typing import List
import os
from .filter import StructureFilter
from pymatgen.core import Structure


def write_cifs(structures: List[Structure], write_directory: str, include_elems: bool):
    """
    Write structures to CIF files

    Args:
        strucs: list of pymatgen Structure objects
        dir: path to directory where CIF files will be written
        include_elems: if True, include single-element structures in the reference set
    """

    if not os.path.isdir(write_directory):
        os.mkdir(write_directory)

    for struc in structures:
        num_elems = struc.composition.elements
        if num_elems == 1:
            if not include_elems:
                continue
        formula = struc.composition.reduced_formula
        try:
            sg = struc.get_space_group_info()[1]
            filepath = os.path.join(write_directory, f"{formula}_{sg}.cif")
            struc.to(filename=filepath, fmt="cif")
        except:
            try:
                print(
                    "%s Space group cant be determined, lowering tolerance"
                    % str(formula)
                )
                sg = struc.get_space_group_info(symprec=0.1, angle_tolerance=5.0)[1]
                filepath = os.path.join(write_directory, f"{formula}_{sg}.cif")
                struc.to(filename=filepath, fmt="cif")
            except:
                print(
                    "%s Space group cant be determined even after lowering tolerance, Setting to None"
                    % str(formula)
                )

    assert (
        len(os.listdir(write_directory)) > 0
    ), "Something went wrong. No reference phases were found."


def main(cif_directory, write_directory, include_single_element_structures=True, enforce_order=False):

    # Get unique structures
    struc_filter = StructureFilter(cif_directory, enforce_order)
    final_refs = struc_filter.filtered_refs

    # Write unique structures (as CIFs) to reference directory
    write_cifs(final_refs, write_directory, include_single_element_structures)
