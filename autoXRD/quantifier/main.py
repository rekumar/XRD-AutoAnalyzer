from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction import xrd
from pymatgen.core import Structure
import numpy as np
from .quantanalysis import QuantAnalysis
from .utils import get_density, get_max_intensity, get_volume


def main(
    spectra_directory,
    spectrum_fname,
    predicted_phases,
    scale_factors,
    min_angle=10.0,
    max_angle=80.0,
    wavelength="CuKa",
):

    analyzer = QuantAnalysis(
        spectra_directory,
        spectrum_fname,
        predicted_phases,
        scale_factors,
        min_angle,
        max_angle,
        wavelength,
    )

    if len(predicted_phases) == 1:
        return [1.0]

    x = np.linspace(min_angle, max_angle, 4501)
    measured_spectrum = analyzer.formatted_spectrum
    angle_sets, intensity_sets = analyzer.scaled_patterns

    I_expec, I_obs, V, dens = [], [], [], []
    for (cmpd, I_set) in zip(predicted_phases, intensity_sets):
        I_obs.append(max(I_set))
        I_expec.append(get_max_intensity(cmpd, min_angle, max_angle))
        V.append(get_volume(cmpd))
        dens.append(get_density(cmpd))

    if len(predicted_phases) == 2:
        c21_ratio = (
            (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1] ** 2 / V[0] ** 2)
        )
        c1 = 1.0 / (1.0 + c21_ratio)
        c2 = 1.0 - c1
        m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2))
        m2 = 1.0 - m1
        return [m1, m2]

    if len(predicted_phases) == 3:
        c21_ratio = (
            (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1] ** 2 / V[0] ** 2)
        )
        c31_ratio = (
            (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2] ** 2 / V[0] ** 2)
        )
        c1 = 1.0 / (1.0 + c21_ratio + c31_ratio)
        c12_ratio = 1.0 / c21_ratio
        c32_ratio = (
            (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2] ** 2 / V[1] ** 2)
        )
        c2 = 1.0 / (c12_ratio + 1.0 + c32_ratio)
        c3 = 1.0 - c1 - c2
        m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3))
        m2 = (dens[1] * c2) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3))
        m3 = 1.0 - m1 - m2
        return [m1, m2, m3]

    if len(predicted_phases) == 4:
        c21_ratio = (
            (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1] ** 2 / V[0] ** 2)
        )
        c31_ratio = (
            (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2] ** 2 / V[0] ** 2)
        )
        c41_ratio = (
            (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3] ** 2 / V[0] ** 2)
        )
        c1 = 1.0 / (1.0 + c21_ratio + c31_ratio + c41_ratio)
        c12_ratio = 1.0 / c21_ratio
        c32_ratio = (
            (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2] ** 2 / V[1] ** 2)
        )
        c42_ratio = (
            (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3] ** 2 / V[1] ** 2)
        )
        c2 = 1.0 / (c12_ratio + 1.0 + c32_ratio + c42_ratio)
        c13_ratio = 1.0 / c31_ratio
        c23_ratio = (
            (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1] ** 2 / V[2] ** 2)
        )
        c43_ratio = (
            (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3] ** 2 / V[2] ** 2)
        )
        c3 = 1.0 / (c13_ratio + c23_ratio + 1.0 + c43_ratio)
        c4 = 1.0 - c1 - c2 - c3
        m1 = (dens[0] * c1) / (
            (dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4)
        )
        m2 = (dens[1] * c2) / (
            (dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4)
        )
        m3 = (dens[2] * c3) / (
            (dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4)
        )
        m4 = 1.0 - m1 - m2 - m3
        return [m1, m2, m3, m4]

    if len(predicted_phases) == 5:
        c21_ratio = (
            (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1] ** 2 / V[0] ** 2)
        )
        c31_ratio = (
            (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2] ** 2 / V[0] ** 2)
        )
        c41_ratio = (
            (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3] ** 2 / V[0] ** 2)
        )
        c51_ratio = (
            (I_obs[4] / I_obs[0]) * (I_expec[0] / I_expec[4]) * (V[4] ** 2 / V[0] ** 2)
        )
        c1 = 1.0 / (1.0 + c21_ratio + c31_ratio + c41_ratio + c51_ratio)
        c12_ratio = 1.0 / c21_ratio
        c32_ratio = (
            (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2] ** 2 / V[1] ** 2)
        )
        c42_ratio = (
            (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3] ** 2 / V[1] ** 2)
        )
        c52_ratio = (
            (I_obs[4] / I_obs[1]) * (I_expec[1] / I_expec[4]) * (V[4] ** 2 / V[1] ** 2)
        )
        c2 = 1.0 / (c12_ratio + 1.0 + c32_ratio + c42_ratio + c52_ratio)
        c13_ratio = 1.0 / c31_ratio
        c23_ratio = (
            (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1] ** 2 / V[2] ** 2)
        )
        c43_ratio = (
            (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3] ** 2 / V[2] ** 2)
        )
        c53_ratio = (
            (I_obs[4] / I_obs[2]) * (I_expec[2] / I_expec[4]) * (V[4] ** 2 / V[2] ** 2)
        )
        c3 = 1.0 / (c13_ratio + c23_ratio + 1.0 + c43_ratio + c53_ratio)
        c14_ratio = 1.0 / c41_ratio
        c24_ratio = (
            (I_obs[1] / I_obs[3]) * (I_expec[3] / I_expec[1]) * (V[1] ** 2 / V[3] ** 2)
        )
        c609_ratio = (
            (I_obs[2] / I_obs[3]) * (I_expec[3] / I_expec[2]) * (V[2] ** 2 / V[3] ** 2)
        )
        c54_ratio = (
            (I_obs[4] / I_obs[3]) * (I_expec[3] / I_expec[4]) * (V[4] ** 2 / V[3] ** 2)
        )
        c4 = 1.0 / (c14_ratio + c24_ratio + 1.0 + c34_ratio + c54_ratio)
        c5 = 1.0 - c1 - c2 - c3 - c4
        m1 = (dens[0] * c1) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
        )
        m2 = (dens[1] * c2) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
        )
        m3 = (dens[2] * c3) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
        )
        m4 = (dens[3] * c4) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
        )
        m5 = 1.0 - m1 - m2 - m3 - m4
        return [m1, m2, m3, m4, m5]

    if len(predicted_phases) == 6:
        c21_ratio = (
            (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1] ** 2 / V[0] ** 2)
        )
        c31_ratio = (
            (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2] ** 2 / V[0] ** 2)
        )
        c41_ratio = (
            (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3] ** 2 / V[0] ** 2)
        )
        c51_ratio = (
            (I_obs[4] / I_obs[0]) * (I_expec[0] / I_expec[4]) * (V[4] ** 2 / V[0] ** 2)
        )
        c61_ratio = (
            (I_obs[5] / I_obs[0]) * (I_expec[0] / I_expec[5]) * (V[5] ** 2 / V[0] ** 2)
        )
        c1 = 1.0 / (1.0 + c21_ratio + c31_ratio + c41_ratio + c51_ratio + c61_ratio)
        c12_ratio = 1.0 / c21_ratio
        c32_ratio = (
            (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2] ** 2 / V[1] ** 2)
        )
        c42_ratio = (
            (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3] ** 2 / V[1] ** 2)
        )
        c52_ratio = (
            (I_obs[4] / I_obs[1]) * (I_expec[1] / I_expec[4]) * (V[4] ** 2 / V[1] ** 2)
        )
        c62_ratio = (
            (I_obs[5] / I_obs[1]) * (I_expec[1] / I_expec[5]) * (V[5] ** 2 / V[1] ** 2)
        )
        c2 = 1.0 / (c12_ratio + 1.0 + c32_ratio + c42_ratio + c52_ratio + c62_ratio)
        c13_ratio = 1.0 / c31_ratio
        c23_ratio = (
            (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1] ** 2 / V[2] ** 2)
        )
        c43_ratio = (
            (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3] ** 2 / V[2] ** 2)
        )
        c53_ratio = (
            (I_obs[4] / I_obs[2]) * (I_expec[2] / I_expec[4]) * (V[4] ** 2 / V[2] ** 2)
        )
        c63_ratio = (
            (I_obs[5] / I_obs[2]) * (I_expec[2] / I_expec[5]) * (V[5] ** 2 / V[2] ** 2)
        )
        c3 = 1.0 / (c13_ratio + c23_ratio + 1.0 + c43_ratio + c53_ratio + c63_ratio)
        c14_ratio = 1.0 / c41_ratio
        c24_ratio = (
            (I_obs[1] / I_obs[3]) * (I_expec[3] / I_expec[1]) * (V[1] ** 2 / V[3] ** 2)
        )
        c34_ratio = (
            (I_obs[2] / I_obs[3]) * (I_expec[3] / I_expec[2]) * (V[2] ** 2 / V[3] ** 2)
        )
        c54_ratio = (
            (I_obs[4] / I_obs[3]) * (I_expec[3] / I_expec[4]) * (V[4] ** 2 / V[3] ** 2)
        )
        c64_ratio = (
            (I_obs[5] / I_obs[3]) * (I_expec[3] / I_expec[5]) * (V[5] ** 2 / V[3] ** 2)
        )
        c4 = 1.0 / (c14_ratio + c24_ratio + 1.0 + c34_ratio + c54_ratio + c64_ratio)
        c15_ratio = 1.0 / c51_ratio
        c25_ratio = (
            (I_obs[1] / I_obs[4]) * (I_expec[4] / I_expec[1]) * (V[1] ** 2 / V[4] ** 2)
        )
        c35_ratio = (
            (I_obs[2] / I_obs[4]) * (I_expec[4] / I_expec[2]) * (V[2] ** 2 / V[4] ** 2)
        )
        c45_ratio = (
            (I_obs[3] / I_obs[4]) * (I_expec[4] / I_expec[3]) * (V[3] ** 2 / V[4] ** 2)
        )
        c65_ratio = (
            (I_obs[5] / I_obs[4]) * (I_expec[4] / I_expec[5]) * (V[5] ** 2 / V[4] ** 2)
        )
        c5 = 1.0 / (c15_ratio + c25_ratio + 1.0 + c35_ratio + c45_ratio + c65_ratio)
        c6 = 1.0 - c1 - c2 - c3 - c4 - c5
        m1 = (dens[0] * c1) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
            + dens[5] * c6
        )
        m2 = (dens[1] * c2) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
            + dens[5] * c6
        )
        m3 = (dens[2] * c3) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
            + dens[5] * c6
        )
        m4 = (dens[3] * c4) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
            + dens[5] * c6
        )
        m5 = (dens[4] * c5) / (
            (dens[0] * c1)
            + (dens[1] * c2)
            + (dens[2] * c3)
            + (dens[3] * c4)
            + (dens[4] * c5)
            + dens[5] * c6
        )
        m6 = 1.0 - m1 - m2 - m3 - m4 - m5
        return [m1, m2, m3, m4, m5, m6]
