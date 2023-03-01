from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction import xrd
from pymatgen.core import Structure


def get_max_intensity(ref_phase, min_angle, max_angle, ref_dir="References"):
    """
    Returns:
        Retrieve maximum intensity for raw (non-scaled) pattern of ref_phase.
    """

    calculator = xrd.XRDCalculator()

    struct = Structure.from_file("%s/%s" % (ref_dir, ref_phase))

    pattern = calculator.get_pattern(
        struct, two_theta_range=(min_angle, max_angle), scaled=False
    )
    angles = pattern.x
    intensities = pattern.y

    return max(intensities)


def get_volume(ref_phase, ref_dir="References"):
    """
    Get unit cell volume of ref_phase.
    """

    struct = Structure.from_file("%s/%s" % (ref_dir, ref_phase))
    return struct.volume


def get_density(ref_phase, ref_dir="References"):
    """
    Get mass density of ref_phase.
    """

    struct = Structure.from_file("%s/%s" % (ref_dir, ref_phase))

    mass = 0
    for site in struct:
        elem_dict = site.species.remove_charges().as_dict()
        for elem_key in elem_dict.keys():
            # Take into account occupancies and species
            mass += elem_dict[elem_key] * Element(elem_key).atomic_mass

    return mass / struct.volume
