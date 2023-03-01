import tensorflow as tf
import numpy as np
import os
from .phase_identifier import PhaseIdentifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(1)
tf.random.set_seed(1)


def merge_results(results, cutoff, max_phases):
    """
    Aggregate XRD + PDF predictions through an ensemble approach
    whereby each phase is weighted by its confidence.

    xrd: dictionary containing predictions from XRD analysis
    pdf: dictionary containing predictions from PDF analysis
    cutoff: minimum confidence (%) to include in final predictions
    max_phases: maximum no. of phases to include in final predictions
    """

    # First, sort by filename to ensure consistent ordering
    zipped_xrd = zip(
        results["XRD"]["filenames"],
        results["XRD"]["phases"],
        results["XRD"]["confs"],
        results["XRD"]["backup_phases"],
        results["XRD"]["scale_factors"],
        results["XRD"]["reduced_spectra"],
    )
    zipped_pdf = zip(
        results["PDF"]["filenames"],
        results["PDF"]["phases"],
        results["PDF"]["confs"],
        results["PDF"]["backup_phases"],
        results["PDF"]["scale_factors"],
        results["PDF"]["reduced_spectra"],
    )
    sorted_xrd = sorted(zipped_xrd, key=lambda x: x[0])
    sorted_pdf = sorted(zipped_pdf, key=lambda x: x[0])
    (
        results["XRD"]["filenames"],
        results["XRD"]["phases"],
        results["XRD"]["confs"],
        results["XRD"]["backup_phases"],
        results["XRD"]["scale_factors"],
        results["XRD"]["reduced_spectra"],
    ) = list(zip(*sorted_xrd))
    (
        results["PDF"]["filenames"],
        results["PDF"]["phases"],
        results["PDF"]["confs"],
        results["PDF"]["backup_phases"],
        results["PDF"]["scale_factors"],
        results["PDF"]["reduced_spectra"],
    ) = list(zip(*sorted_pdf))

    # Double-check consistent length and order
    assert len(results["XRD"]["filenames"]) == len(
        results["PDF"]["filenames"]
    ), "XRD and PDF prediction are not the same length. Something went wrong."
    for i, filename in enumerate(results["XRD"]["filenames"]):
        assert (
            filename == results["XRD"]["filenames"][i]
        ), "Mismatch between order of XRD and PDF predictions. Something went wrong"

    # Concatenate phases and confidences
    results["All"] = {}
    results["All"]["phases"] = []
    for l1, l2 in zip(results["XRD"]["phases"], results["PDF"]["phases"]):
        results["All"]["phases"].append(list(l1) + list(l2))
    results["All"]["confs"] = []
    for l1, l2 in zip(results["XRD"]["confs"], results["PDF"]["confs"]):
        results["All"]["confs"].append(list(l1) + list(l2))

    # Aggregate XRD and PDF predictions into merged dictionary
    results["Merged"] = {}
    results["Merged"]["phases"] = []
    results["Merged"]["confs"] = []
    results["Merged"]["backup_phases"] = []
    results["Merged"]["scale_factors"] = []
    results["Merged"]["filenames"] = results["XRD"]["filenames"]
    results["Merged"]["reduced_spectra"] = results["XRD"]["reduced_spectra"]
    for (
        phases,
        confs,
        xrd_phases,
        xrd_confs,
        pdf_phases,
        pdf_confs,
        xrd_backup_phases,
        pdf_backup_phases,
        xrd_scale_factors,
        pdf_scale_factors,
    ) in zip(
        results["All"]["phases"],
        results["All"]["confs"],
        results["XRD"]["phases"],
        results["XRD"]["confs"],
        results["PDF"]["phases"],
        results["PDF"]["confs"],
        results["XRD"]["backup_phases"],
        results["PDF"]["backup_phases"],
        results["XRD"]["scale_factors"],
        results["PDF"]["scale_factors"],
    ):

        # Allocate confidences for each phase
        avg_soln = {}
        for cmpd, cf in zip(phases, confs):
            if cmpd not in avg_soln.keys():
                avg_soln[cmpd] = [cf]
            else:
                avg_soln[cmpd].append(cf)

        # Average over confidences for each phase
        unique_phases, avg_confs = [], []
        for cmpd in avg_soln.keys():
            unique_phases.append(cmpd)
            num_zeros = 2 - len(avg_soln[cmpd])
            avg_soln[cmpd] += [0.0] * num_zeros
            avg_confs.append(np.mean(avg_soln[cmpd]))

        # Sort by confidence
        info = zip(unique_phases, avg_confs)
        info = sorted(info, key=lambda x: x[1])
        info.reverse()

        # Get all unique phases below max no.
        unique_phases, unique_confs = [], []
        for cmpd, cf in info:
            if (len(unique_phases) < max_phases) and (cf > cutoff):
                unique_phases.append(cmpd)
                unique_confs.append(cf)

        # Collect final backups and scale factors
        unique_backups, unique_scales = [], []
        for cmpd in unique_phases:
            if cmpd in xrd_phases:
                xrd_ind = xrd_phases.index(cmpd)
                xrd_conf = xrd_confs[xrd_ind]
            else:
                xrd_conf = 0.0
            if cmpd in pdf_phases:
                pdf_ind = pdf_phases.index(cmpd)
                pdf_conf = pdf_confs[pdf_ind]
            else:
                pdf_conf = 0.0
            if xrd_conf >= pdf_conf:
                unique_backups.append(xrd_backup_phases[xrd_ind])
                unique_scales.append(xrd_scale_factors[xrd_ind])
            else:
                unique_backups.append(pdf_backup_phases[pdf_ind])
                unique_scales.append(pdf_scale_factors[pdf_ind])

        results["Merged"]["phases"].append(unique_phases)
        results["Merged"]["confs"].append(unique_confs)
        results["Merged"]["backup_phases"].append(unique_backups)
        results["Merged"]["scale_factors"].append(unique_scales)

    return results["Merged"]


def main(
    spectra_directory,
    reference_directory,
    max_phases=3,
    cutoff_intensity=10,
    min_conf=10.0,
    wavelength="CuKa",
    min_angle=10.0,
    max_angle=80.0,
    parallel=True,
    model_path="Model.h5",
    is_pdf=False,
):

    phase_id = PhaseIdentifier(
        spectra_directory,
        reference_directory,
        max_phases,
        cutoff_intensity,
        min_conf,
        wavelength,
        min_angle,
        max_angle,
        parallel,
        model_path,
        is_pdf,
    )

    (
        spectrum_names,
        predicted_phases,
        confidences,
        backup_phases,
        scale_factors,
        reduced_spectra,
    ) = phase_id.all_predictions

    return (
        spectrum_names,
        predicted_phases,
        confidences,
        backup_phases,
        scale_factors,
        reduced_spectra,
    )
