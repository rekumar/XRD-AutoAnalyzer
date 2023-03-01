import os
from .filter import StructureFilter

def write_cifs(unique_strucs, dir, include_elems):
    """
    Write structures to CIF files

    Args:
        strucs: list of pymatgen Structure objects
        dir: path to directory where CIF files will be written
    """

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for struc in unique_strucs:
        num_elems = struc.composition.elements
        if num_elems == 1:
            if not include_elems:
                continue
        formula = struc.composition.reduced_formula
        f = struc.composition.reduced_formula
        try:
            sg = struc.get_space_group_info()[1]
            filepath = "%s/%s_%s.cif" % (dir, f, sg)
            struc.to(filename=filepath, fmt="cif")
        except:
            try:
                print("%s Space group cant be determined, lowering tolerance" % str(f))
                sg = struc.get_space_group_info(symprec=0.1, angle_tolerance=5.0)[1]
                filepath = "%s/%s_%s.cif" % (dir, f, sg)
                struc.to(filename=filepath, fmt="cif")
            except:
                print(
                    "%s Space group cant be determined even after lowering tolerance, Setting to None"
                    % str(f)
                )

    assert (
        len(os.listdir(dir)) > 0
    ), "Something went wrong. No reference phases were found."


def main(cif_directory, ref_directory, include_elems=True, enforce_order=False):

    # Get unique structures
    struc_filter = StructureFilter(cif_directory, enforce_order)
    final_refs = struc_filter.filtered_refs

    # Write unique structures (as CIFs) to reference directory
    write_cifs(final_refs, ref_directory, include_elems)
