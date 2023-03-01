from .generator import SolidSolnsGen


def main(reference_directory):

    ns_generator = SolidSolnsGen(reference_directory)

    solid_solns = ns_generator.all_solid_solns

    for struc in solid_solns:

        # Name file according to its composition and space group
        filepath = "%s/%s_%s.cif" % (
            reference_directory,
            struc.composition.reduced_formula,
            struc.get_space_group_info()[1],
        )

        # Do not write if a known stoichiometric reference phase already exists
        if filepath.split("/")[1] not in os.listdir(reference_directory):
            struc.to(filename=filepath, fmt="cif")
