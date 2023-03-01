import math
from scipy.ndimage import gaussian_filter1d
from typing import List, Union, Tuple
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from scipy.signal import resample


def get_radiation_wavelength(radiation_source: str) -> float:
    """Get the wavelength of the radiation source in angstroms. We use the sources enumerated in pymatgen.analysis.diffraction.xrd.WAVELENGTHS

    Args:
        radiation_source (str): Either the name of a common radiation source (CuKa, CuKa2, etc), or the wavelength in angstroms.

    Returns:
        float: Wavelength of radiation source in angstroms.
    """

    if isinstance(radiation_source, str):
        if radiation_source not in WAVELENGTHS:
            raise ValueError(
                f"Unrecognized emission line name: {radiation_source}. "
                "Either pass the wavelength (in angstroms) directly as a float, or use one of the following: "
                f"{list(WAVELENGTHS.keys())}"
            )
        return WAVELENGTHS[radiation_source]
    else:
        radiation_source = float(radiation_source)
        if radiation_source < 0:
            raise ValueError("Wavelength must be a positive number.")
        return radiation_source


def convert_angle(
    self,
    two_theta: float,
    original_wavelength_angstroms: float,
    target_wavelength_angstroms: float = 1.5406,
) -> Union[float, None]:
    """Convert two-theta into Cu K-alpha radiation.


    Args:
        two_theta (float): Diffracting angle at original wavelength.
        original_wavelength_angstroms (float): Wavelength (angstroms) of radiation used to collect data.
        target_wavelength_angstroms (float, optional): Wavelength (angstroms) to of radiation to adjdust angle to. Defaults to 1.5406.

    Returns:
        float: Angle at target wavelength. Returns None if angle would be outside of 0-90 range (arcsin > 1)
    """

    orig_theta = math.radians(two_theta / 2.0)

    original_wavelength_angstroms
    ratio_lambda = target_wavelength_angstroms / original_wavelength_angstroms

    asin_argument = ratio_lambda * math.sin(orig_theta)

    # Curtail two-theta range if needed to avoid domain errors
    if asin_argument <= 1:
        new_theta = math.degrees(math.asin(ratio_lambda * math.sin(orig_theta)))
        return 2 * new_theta

    return None  # Angle would be outside of 0-90 range, invalid arcsin argument #TODO - handle this better


def simulated_domain_size_stddev(
    self, two_theta: float, domain_size_nm: float, wavelength_angstroms: float = 1.5406
):
    """
    Used to approximate the effect of diffracting domain size on the peak widths of simulated XRD patterns. Uses the Scherrer equation to calculate the FWHM of a peak, then converts to a standard deviation for a gaussian kernel that can be used to convolve a theoretical pattern.

    Args:
        two_theta: angle in two theta
        domain_size_nm: domain size in nm
        wavelength_angstroms: wavelength of radiation (angstroms) used to collect data. Defaults to Cu K-alpha (1.5406 ang)
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.9  ## shape factor
    wavelength_nm = wavelength_angstroms * 0.1  ## angstrom to nm
    theta = np.radians(two_theta / 2.0)  ## Bragg angle in radians
    beta = (K * wavelength_nm) / (np.cos(theta) * domain_size_nm)  # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
    return sigma**2


def get_stick_pattern(
    self, structure: Structure, twotheta_min: float, twotheta_max: float
) -> Tuple(np.ndarray, np.ndarray):

    pattern = self.calculator.get_pattern(
        structure, two_theta_range=(twotheta_min, twotheta_max)
    )
    angles = pattern.x
    intensities = pattern.y

    return angles, intensities


def generate_pattern(
    structure: Structure,
    twotheta: List[float],
    normalize: bool = True,
    domain_size_nm: float = 25.0,
    wavelength_angstroms: float = 1.5406,
) -> np.ndarray:

    twotheta = np.array(twotheta)
    twotheta_step_size = np.abs(
        twotheta[1] - twotheta[0]
    )  # assume step size is constant
    signals = np.zeros_like(twotheta)

    # Start with stick pattern from pymatgen. For each stick (peak), we convolute with a gaussian kernel to simulate Scherrer broadening from diffracting domain size, then sum all peaks to get the final pattern.

    stick_pattern = XRDCalculator().get_pattern(
        structure, two_theta_range=(twotheta.min(), twotheta.max())
    )

    peak_broadening_matrix = np.zeros((len(stick_pattern.x), len(twotheta)))

    for row_idx, peak_center in enumerate(stick_pattern.x):
        stick_col_idx = np.argmin(np.abs(twotheta - peak_center))
        peak_broadening_matrix[row_idx, stick_col_idx] = stick_pattern.y[row_idx]

        std_dev = simulated_domain_size_stddev(
            two_theta=peak_center,
            domain_size_nm=domain_size_nm,
            wavelength_angstroms=wavelength_angstroms,
        )

        # Convolution is expressed in matrix steps, not angle
        peak_broadening_matrix[row_idx, :] = gaussian_filter1d(
            input=peak_broadening_matrix[row_idx, :],
            sigma=np.sqrt(std_dev) / twotheta_step_size,
            mode="constant",
        )

    # Combine signals
    signal = np.sum(peak_broadening_matrix, axis=0)

    # Normalize signal
    if normalize:
        signal /= np.max(signal)
        signal *= 100

    return signal


def XRDtoPDF(
    twotheta: List[float], xrd: List[float], wavelength_angstroms: float = 1.5406
) -> List[float]:

    thetas = twotheta / 2
    Q = np.array(
        [
            4 * math.pi * math.sin(math.radians(theta)) / wavelength_angstroms
            for theta in thetas
        ]
    )
    S = np.array(xrd).flatten()

    pdf = []
    R = np.linspace(1, 40, 1000)  # Only 1000 used to reduce compute time
    integrand = Q * S * np.sin(Q * R[:, np.newaxis])

    pdf = 2 * np.trapz(integrand, Q) / math.pi
    pdf = list(resample(pdf, len(twotheta)))

    return pdf
