import matplotlib.pyplot as plt
import numpy as np
from .plotter import SpectrumPlotter
from .utils import XRDtoPDF

def main(
    spectra_directory,
    spectrum_fname,
    predicted_phases,
    scale_factors,
    reduced_spectrum,
    min_angle=10.0,
    max_angle=80.0,
    wavelength="CuKa",
    save=False,
    show_reduced=False,
    inc_pdf=False,
):

    spec_plot = SpectrumPlotter(
        spectra_directory,
        spectrum_fname,
        predicted_phases,
        scale_factors,
        min_angle,
        max_angle,
        wavelength,
    )

    x = np.linspace(min_angle, max_angle, 4501)
    measured_spectrum = spec_plot.formatted_spectrum
    angle_sets, intensity_sets = spec_plot.scaled_patterns

    plt.figure()

    plt.plot(x, measured_spectrum, "b-", label="Measured: %s" % spectrum_fname)

    phase_names = [fname[:-4] for fname in predicted_phases]  # remove .cif
    color_list = ["g", "r", "m", "k", "c"]
    i = 0
    for (angles, intensities, phase) in zip(angle_sets, intensity_sets, phase_names):
        for (xv, yv) in zip(angles, intensities):
            plt.vlines(xv, 0, yv, color=color_list[i], linewidth=2.5)
        plt.plot([0], [0], color=color_list[i], label="Predicted: %s" % phase)
        i += 1

    if show_reduced:
        plt.plot(
            x,
            reduced_spectrum,
            color="orange",
            linestyle="dashed",
            label="Reduced spectrum",
        )

    plt.xlim(min_angle, max_angle)
    plt.ylim(0, 105)
    plt.legend(prop={"size": 16})
    plt.xlabel(r"2$\Theta$", fontsize=16, labelpad=12)
    plt.ylabel("Intensity", fontsize=16, labelpad=12)

    if save:
        savename = "%s.png" % spectrum_fname.split(".")[0]
        plt.tight_layout()
        plt.savefig(savename, dpi=400)
        plt.close()

    else:
        plt.show()

    plt.close()

    if inc_pdf:

        r, measured_pdf = XRDtoPDF(measured_spectrum, min_angle, max_angle)

        plt.figure()

        plt.plot(r, measured_pdf, "b-", label="Measured: %s" % spectrum_fname)

        phase_names = [fname[:-4] for fname in predicted_phases]  # remove .cif
        color_list = ["g", "r", "m", "k", "c"]
        i = 0
        for (angles, intensities, phase) in zip(
            angle_sets, intensity_sets, phase_names
        ):
            ys = spec_plot.get_cont_profile(angles, intensities)
            r, ref_pdf = XRDtoPDF(ys, min_angle, max_angle)
            plt.plot(
                r,
                ref_pdf,
                color=color_list[i],
                linestyle="dashed",
                label="Predicted: %s" % phase,
            )
            i += 1

        plt.xlim(1, 30)
        plt.legend(prop={"size": 16})
        plt.xlabel(r"r (Å)", fontsize=16, labelpad=12)
        plt.ylabel("G(r)", fontsize=16, labelpad=12)

        if save:
            savename = "%s_PDF.png" % spectrum_fname.split(".")[0]
            plt.tight_layout()
            plt.savefig(savename, dpi=400)
            plt.close()

        else:
            plt.show()
