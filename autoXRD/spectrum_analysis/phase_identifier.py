from multiprocessing import Pool, Manager
import multiprocessing
from tqdm import tqdm
import numpy as np
import os

class PhaseIdentifier(object):
    """
    Class used to identify phases from a given set of xrd spectra
    """

    def __init__(self, spectra_directory, reference_directory, max_phases, cutoff_intensity, min_conf, wavelength,
        min_angle=10.0, max_angle=80.0, parallel=True, model_path='Model.h5', is_pdf=False):
        """
        Args:
            spectra_dir: path to directory containing the xrd
                spectra to be analyzed
            reference_directory: path to directory containing
                the reference phases
        """

        self.num_cpu = multiprocessing.cpu_count()
        self.spectra_dir = spectra_directory
        self.ref_dir = reference_directory
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.min_conf = min_conf
        self.wavelen = wavelength
        self.parallel = parallel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.model_path = model_path
        self.is_pdf = is_pdf

    @property
    def all_predictions(self):
        """
        Returns:
            spectrum_names: filenames of spectra being classified
            predicted_phases: a list of the predicted phases in the mixture
            confidences: the associated confidence with the prediction above
        """

        reference_phases = sorted(os.listdir(self.ref_dir))
        spectrum_filenames = os.listdir(self.spectra_dir)
        spectrum_filenames = [fname for fname in spectrum_filenames if fname[0] != '.']

        if self.parallel:
            with Manager() as manager:
                pool = Pool(self.num_cpu)
                if self.is_pdf:
                    print('Running PDF analysis')
                else:
                    print('Running XRD analysis')
                all_info = list(tqdm(pool.imap(self.classify_mixture, spectrum_filenames),
                    total=len(spectrum_filenames)))

        else:
            all_info = []
            for filename in spectrum_filenames:
                all_info.append(self.classify_mixture(filename))

        spectrum_fnames = [info[0] for info in all_info]
        predicted_phases = [info[1] for info in all_info]
        confidences = [info[2] for info in all_info]
        backup_phases = [info[3] for info in all_info]
        scale_factors = [info[4] for info in all_info]
        spectra = [info[5][-1] for info in all_info]

        return spectrum_fnames, predicted_phases, confidences, backup_phases, scale_factors, spectra

    def classify_mixture(self, spectrum_fname):
        """
        Args:
            fname: filename string of the spectrum to be classified
        Returns:
            fname: filename, same as in Args
            predicted_set: string of compounds predicted by phase ID algo
            max_conf: confidence associated with the prediction
        """

        total_confidence, all_predictions = [], []
        tabulate_conf, predicted_cmpd_set = [], []

        spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases, self.cutoff,
            self.min_conf, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=self.max_angle,
            model_path=self.model_path, is_pdf=self.is_pdf)

        mixtures, confidences, backup_mixtures, scalings, spectra = spec_analysis.suspected_mixtures

        # If classification is non-trival, identify most probable mixture
        if any(confidences):
            avg_conf = [np.mean(conf) for conf in confidences]
            max_conf_ind = np.argmax(avg_conf)
            final_confidences = [round(100*val, 2) for val in confidences[max_conf_ind]]
            scaling_constants = [round(val, 3) for val in scalings[max_conf_ind]]
            specs = spectra[max_conf_ind]
            predicted_set = [fname[:-4] for fname in mixtures[max_conf_ind]]
            backup_set = []
            for ph in backup_mixtures[max_conf_ind]:
                if 'cif' in str(ph):
                    backup_set.append(ph[:-4])
                else:
                    backup_set.append('None')

        # Otherwise, return None
        else:
            final_confidences = [0.0]
            scaling_constants = [0.0]
            specs = [[]]
            predicted_set = ['None']
            backup_set = ['None']

        return [spectrum_fname, predicted_set, final_confidences, backup_set, scaling_constants, specs]
