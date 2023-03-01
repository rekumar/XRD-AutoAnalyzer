from typing import List, Optional, Union
from scipy.signal import filtfilt, resample
from scipy import interpolate as ip
from pymatgen.core import Structure
from skimage import restoration
import tensorflow as tf
from pyts import metrics
import numpy as np
import warnings
import os
from autoXRD.pattern_analysis.utils import (
    convert_angle,
    generate_pattern,
    get_radiation_wavelength,
    XRDtoPDF,
    scale_spectrum,
    strip_spectrum,
    smooth_spectrum,
)
from autoXRD.cnn.dropout import CustomDropout, KerasDropoutPrediction


class PatternAnalyzer(object):
    """
    Class used to process and classify xrd patterns using a pretrained AutoXRD model.
    """

    def __init__(
        self,
        max_phases: int,
        cutoff_intensity: float,
        min_conf: float = 25.0,
        wavelength: Union[str, float] = "CuKa",
        min_angle: float = 10.0,
        max_angle: float = 80.0,
        model_path: str = "Model.h5",
        is_pdf: bool = False,
    ):
        """
        Args:
            spectrum_fname: name of file containing the
                xrd spectrum (in xy format)
            reference_dir: path to directory containing the
                reference phases (CIF files)
            wavelen: wavelength used for diffraction (angstroms).
                Defaults to Cu K-alpha radiation (1.5406 angstroms).
        """
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.min_conf = min_conf
        self.wavelen = get_radiation_wavelength(wavelength)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.model_path = model_path
        self.is_pdf = is_pdf
        self._load_model()

    def _load_model(self, model_path: Optional[str] = None):
        if model_path:
            self.model_path = model_path
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"CustomDropout": CustomDropout},
            compile=False,
        )
        self.kdp = KerasDropoutPrediction(self.model)

    def predict(self, twotheta: List[float], intensity: List[float]):
        """
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        formatted_intensity = self._format_spectrum(twotheta, intensity)

        (
            prediction_list,
            confidence_list,
            backup_list,
            scale_list,
            spec_list,
        ) = self._enumerate_routes(formatted_intensity)

        return prediction_list, confidence_list, backup_list, scale_list, spec_list

    def _format_spectrum(
        self, twotheta: List[float], intensity: List[float]
    ) -> np.ndarray:
        """
        Cleans up a measured spectrum and format it such that it
        is directly readable by the CNN.

        Args:
            twotheta: List of two-theta values
            intensity: List of intensity values
        Returns:
            ys: Processed pattern intensity in 4501x1 form.
        """

        ## Load data

        x = twotheta
        y = intensity

        ## Convert to Cu K-alpha radiation if needed
        if self.wavelen != get_radiation_wavelength("CuKa"):
            Cu_x, Cu_y = [], []
            for (two_thet, intens) in zip(x, y):
                scaled_x = convert_angle(
                    two_theta=two_thet,
                    original_wavelength_angstroms=self.wavelen,
                    target_wavelength_angstroms=1.5406,
                )
                if scaled_x is not None:
                    Cu_x.append(scaled_x)
                    Cu_y.append(intens)
            x, y = Cu_x, Cu_y

        # Allow some tolerance (0.2 degrees) in the two-theta range
        if (min(x) > self.min_angle) and np.isclose(min(x), self.min_angle, atol=0.2):
            x = np.concatenate([np.array([self.min_angle]), x])
            y = np.concatenate([np.array([y[0]]), y])
        if (max(x) < self.max_angle) and np.isclose(max(x), self.max_angle, atol=0.2):
            x = np.concatenate([x, np.array([self.max_angle])])
            y = np.concatenate([y, np.array([y[-1]])])

        # Otherwise, raise an assertion error
        assert (min(x) <= self.min_angle) and (
            max(x) >= self.max_angle
        ), """
               Measured spectrum does not span the specified two-theta range!
               Either use a broader spectrum or change the two-theta range via
               the --min_angle and --max_angle arguments."""

        ## Fit to 4,501 values as to be compatible with CNN
        f = ip.CubicSpline(x, y)
        xs = np.linspace(self.min_angle, self.max_angle, 4501)
        ys = f(xs)

        ## Smooth out noise
        ys = smooth_spectrum(ys)

        ## Normalize from 0 to 255
        ys = np.array(ys) - min(ys)
        ys = list(255 * np.array(ys) / max(ys))

        # Subtract background
        background = restoration.rolling_ball(ys, radius=800)
        ys = np.array(ys) - np.array(background)

        ## Normalize from 0 to 100
        ys = np.array(ys) - min(ys)
        ys = list(100 * np.array(ys) / max(ys))

        return ys

    def _get_reduced_pattern(self, predicted_cmpd, orig_y, last_normalization=1.0):
        """
        Subtract a phase that has already been identified from a given XRD spectrum.
        If all phases have already been identified, halt the iteration.

        Args:
            predicted_cmpd: phase that has been identified
            orig_y: measured spectrum including the phase the above phase
            last_normalization: normalization factor used to scale the previously stripped
                spectrum to 100 (required by the CNN). This is necessary to determine the
                magnitudes of intensities relative to the initially measured pattern.
            cutoff: the % cutoff used to halt the phase ID iteration. If all intensities are
                below this value in terms of the originally measured maximum intensity, then
                the code assumes that all phases have been identified.
        Returns:
            stripped_y: new spectrum obtained by subtrating the peaks of the identified phase
            new_normalization: scaling factor used to ensure the maximum intensity is equal to 100
            Or
            If intensities fall below the cutoff, preserve orig_y and return Nonetype
                the for new_normalization constant.
        """

        # Simulate spectrum for predicted compounds
        pred_y = self._generate_pattern(predicted_cmpd)

        # Convert to numpy arrays
        pred_y = np.array(pred_y)
        orig_y = np.array(orig_y)

        # Downsample spectra (helps reduce time for DTW)
        downsampled_res = 0.1  # new resolution: 0.1 degrees
        num_pts = int((self.max_angle - self.min_angle) / downsampled_res)
        orig_y = resample(orig_y, num_pts)
        pred_y = resample(pred_y, num_pts)

        # Calculate window size for DTW
        allow_shifts = 0.75  # Allow shifts up to 0.75 degrees
        window_size = int(allow_shifts * num_pts / (self.max_angle - self.min_angle))

        # Get warped spectrum (DTW)
        distance, path = metrics.dtw(
            pred_y,
            orig_y,
            method="sakoechiba",
            options={"window_size": window_size},
            return_path=True,
        )
        index_pairs = path.transpose()
        warped_spectrum = orig_y.copy()
        for ind1, ind2 in index_pairs:
            distance = abs(ind1 - ind2)
            if distance <= window_size:
                warped_spectrum[ind2] = pred_y[ind1]
            else:
                warped_spectrum[ind2] = 0.0

        # Now, upsample spectra back to their original size (4501)
        warped_spectrum = resample(warped_spectrum, 4501)
        orig_y = resample(orig_y, 4501)

        # Scale warped spectrum so y-values match measured spectrum
        scaled_spectrum, scaling_constant = scale_spectrum(warped_spectrum, orig_y)

        # Subtract scaled spectrum from measured spectrum
        stripped_y = strip_spectrum(scaled_spectrum, orig_y)
        stripped_y = smooth_spectrum(stripped_y)
        stripped_y = np.array(stripped_y) - min(stripped_y)

        # Normalization
        new_normalization = 100 / max(stripped_y)
        actual_intensity = max(stripped_y) / last_normalization

        # Calculate actual scaling constant
        scaling_constant /= last_normalization

        # If intensities fall above cutoff, halt enumeration
        if actual_intensity < self.cutoff:
            is_done = True
        else:
            is_done = False

        stripped_y = new_normalization * stripped_y

        return (
            stripped_y,
            last_normalization * new_normalization,
            scaling_constant,
            is_done,
        )

    def _generate_pattern(self, cmpd):
        """
        Calculate the XRD spectrum of a given compound.

        Args:
            cmpd: filename of the structure file to calculate the spectrum for
        Returns:
            all_I: list of intensities as a function of two-theta
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # don't print occupancy-related warnings
            struct = Structure.from_file("%s/%s" % (self.ref_dir, cmpd))
        twotheta = np.linspace(self.min_angle, self.max_angle, 4501)
        signal = generate_pattern(
            structure=struct,
            twotheta=twotheta,
            normalize=True,
            domain_size_nm=25.0,
            wavelength_angstroms=self.wavelen,
        )

        return signal

    def _enumerate_routes(
        self,
        xrd_spectrum,
        indiv_pred=[],
        indiv_conf=[],
        indiv_backup=[],
        prediction_list=[],
        confidence_list=[],
        backup_list=[],
        is_first=True,
        normalization=1.0,
        indiv_scale=[],
        scale_list=[],
        indiv_spec=[],
        spec_list=[],
    ):
        """
        A branching algorithm designed to explore all suspected mixtures predicted by the CNN.
        For each mixture, the associated phases and probabilities are tabulated.

        Args:
            xrd_spectrum: a numpy array containing the measured spectrum that is to be classified
            kdp: a KerasDropoutPrediction model object
            reference_phases: a list of reference phase strings
            indiv_conf: list of probabilities associated with an individual mixture (one per branch)
            indiv_pred: list of predicted phases in an individual mixture (one per branch)
            confidence_list: a list of averaged probabilities associated with all suspected mixtures
            predictions_list: a list of the phases predicted in all suspected mixtures
            max_phases: the maximum number of phases considered for a single mixture.
                By default, this is set to handle  up tothree-phase patterns. The function is readily
                extended to handle arbitrary many phases. Caution, however, that the computational time
                required will scale exponentially with the number of phases.
            is_first: determines whether this is the first iteration for a given mixture. If it is,
                all global variables will be reset
            normalization: keep track of stripped pattern intensity relative to initial maximum.
                For example, a stripped pattern with half the intensity of hte initial maximum
                should be associated with a normalization constant of 2 (I_0/I_new).
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        # Convert to PDF if specified
        if self.is_pdf:
            twotheta = np.linspace(self.min_angle, self.max_angle, 4501)
            pdf_spectrum = XRDtoPDF(
                twotheta=twotheta, xrd=xrd_spectrum, wavelength_angstroms=self.wavelen
            )

        # Make prediction and confidence lists global so they can be updated recursively
        # If this is the top-level of a new mixture (is_first), reset all variables
        if is_first:
            global updated_pred, updated_conf, updated_backup, updated_scale, updated_spec
            updated_pred, updated_conf, updated_backup, updated_scale, updated_spec = (
                None,
                None,
                None,
                None,
                None,
            )
            prediction_list, confidence_list, backup_list, scale_list = [], [], [], []
            indiv_pred, indiv_conf, indiv_backup, indiv_scale = [], [], [], []
            indiv_spec, spec_list = [], []

        # Make prediction regarding top phases
        if self.is_pdf:
            prediction, num_phases, certanties = self.kdp.predict(
                pdf_spectrum, self.min_conf
            )
        else:
            prediction, num_phases, certanties = self.kdp.predict(
                xrd_spectrum, self.min_conf
            )

        # If no phases are suspected
        if num_phases == 0:

            # If individual predictions have been updated recursively, use them for this iteration
            if "updated_pred" in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf, indiv_scale, indiv_backup, indiv_spec = (
                        updated_pred,
                        updated_conf,
                        updated_scale,
                        updated_backup,
                        updated_spec,
                    )
                    (
                        updated_pred,
                        updated_conf,
                        updated_scale,
                        updated_backup,
                        updated_spec,
                    ) = (None, None, None, None, None)

            confidence_list.append(indiv_conf)
            prediction_list.append(indiv_pred)
            backup_list.append(indiv_backup)
            scale_list.append(indiv_scale)
            spec_list.append(indiv_spec)

        # Explore all phases with a non-trival probability
        for i in range(num_phases):

            # If individual predictions have been updated recursively, use them for this iteration
            if "updated_pred" in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf, indiv_scale, indiv_backup, indiv_spec = (
                        updated_pred,
                        updated_conf,
                        updated_scale,
                        updated_backup,
                        updated_spec,
                    )
                    (
                        updated_pred,
                        updated_conf,
                        updated_scale,
                        updated_backup,
                        updated_spec,
                    ) = (None, None, None, None, None)

            phase_index = np.array(prediction).argsort()[-(i + 1)]
            predicted_cmpd = self.reference_phases[phase_index]

            # If there exists two probable phases
            if num_phases > 1:
                # For 1st most probable phase, choose 2nd most probable as backup
                if i == 0:
                    backup_index = np.array(prediction).argsort()[-(i + 2)]
                # For 2nd most probable phase, choose 1st most probable as backup
                # For 3rd most probable phase, choose 2nd most probable as backup (and so on)
                elif i >= 1:
                    backup_index = np.array(prediction).argsort()[-i]
                backup_cmpd = self.reference_phases[backup_index]
            # If only one phase is suspected, no backups are needed
            else:
                backup_cmpd = None

            # If the predicted phase has already been identified for the mixture, ignore and move on
            if predicted_cmpd in indiv_pred:
                if i == (num_phases - 1):
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    backup_list.append(indiv_backup)
                    scale_list.append(indiv_scale)
                    spec_list.append(indiv_spec)
                    updated_conf, updated_pred = indiv_conf[:-1], indiv_pred[:-1]
                    updated_backup, updated_scale = indiv_backup[:-1], indiv_scale[:-1]
                    updated_spec = indiv_spec[:-1]

                continue

            # Otherwise if phase is new, add to the suspected mixture
            indiv_pred.append(predicted_cmpd)

            # Tabulate the probability associated with the predicted phase
            indiv_conf.append(certanties[i])

            # Tabulate alternative phases
            indiv_backup.append(backup_cmpd)

            # Subtract identified phase from the spectrum
            (
                reduced_spectrum,
                norm,
                scaling_constant,
                is_done,
            ) = self._get_reduced_pattern(
                predicted_cmpd, xrd_spectrum, last_normalization=normalization
            )

            # Record actual spectrum (non-scaled) after peak substraction of known phases
            actual_spectrum = reduced_spectrum / norm
            indiv_spec.append(actual_spectrum)

            # Record scaling constant for each phase (to be used for weight fraction estimates)
            indiv_scale.append(scaling_constant)

            # If all phases have been identified, tabulate mixture and move on to next
            if is_done:
                reduced_spectrum = xrd_spectrum.copy()
                confidence_list.append(indiv_conf)
                prediction_list.append(indiv_pred)
                scale_list.append(indiv_scale)
                backup_list.append(indiv_backup)
                spec_list.append(indiv_spec)
                if i == (num_phases - 1):
                    updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                    updated_backup, updated_scale = indiv_backup[:-2], indiv_scale[:-2]
                    updated_spec = indiv_spec[:-2]
                else:
                    indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                    indiv_backup, indiv_scale = indiv_backup[:-1], indiv_scale[:-1]
                    updated_spec = indiv_spec[:-2]
                continue

            else:
                # If the maximum number of phases has been reached, tabulate mixture and move on to next
                if len(indiv_pred) == self.max_phases:
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    scale_list.append(indiv_scale)
                    backup_list.append(indiv_backup)
                    spec_list.append(indiv_spec)
                    if i == (num_phases - 1):
                        updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                        updated_backup, updated_scale = (
                            indiv_backup[:-2],
                            indiv_scale[:-2],
                        )
                        updated_spec = indiv_spec[:-2]
                    else:
                        indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                        indiv_backup, indiv_scale = indiv_backup[:-1], indiv_scale[:-1]
                        indiv_spec = indiv_spec[:-1]
                    continue

                # Otherwise if more phases are to be explored, recursively enter enumerate_routes with the newly reduced spectrum
                (
                    prediction_list,
                    confidence_list,
                    backup_list,
                    scale_list,
                    spec_list,
                ) = self._enumerate_routes(
                    reduced_spectrum,
                    indiv_pred,
                    indiv_conf,
                    indiv_backup,
                    prediction_list,
                    confidence_list,
                    backup_list,
                    is_first=False,
                    normalization=norm,
                    indiv_scale=indiv_scale,
                    scale_list=scale_list,
                    indiv_spec=indiv_spec,
                    spec_list=spec_list,
                )

        return prediction_list, confidence_list, backup_list, scale_list, spec_list


class DirectoryPatternAnalyzer(PatternAnalyzer):
    """
    Class for analyzing a directory of patterns.

    Attributes
    ----------
    directory : str
        Path to directory containing patterns to be analyzed.
    reference_phases : list
        List of phases to be used for analysis.
    max_phases : int
        Maximum number of phases to be considered in a mixture.
    """

    def __init__(
        self,
        spectra_dir,
        max_phases,
        cutoff_intensity,
        min_conf=25.0,
        wavelength="CuKa",
        reference_dir="References",
        min_angle=10.0,
        max_angle=80.0,
        model_path="Model.h5",
        is_pdf=False,
    ):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing patterns to be analyzed.
        reference_phases : list
            List of phases to be used for analysis.
        max_phases : int
            Maximum number of phases to be considered in a mixture.
        """
        super().__init__(
            max_phases=max_phases,
            cutoff_intensity=cutoff_intensity,
            min_conf=min_conf,
            wavelength=wavelength,
            min_angle=min_angle,
            max_angle=max_angle,
            model_path=model_path,
            is_pdf=is_pdf,
        )

        self.spectra_dir = spectra_dir
        self.ref_dir = reference_dir

    @property
    def reference_phases(self):
        return sorted(os.listdir(self.ref_dir))

    def predict(self, filename: str):
        """
        Predicts the phases present in a given spectrum.

        Parameters
        ----------
        filename : str
            Name of file to be analyzed.

        Returns
        -------
        prediction_list : list
            List of all enumerated mixtures.
        confidence_list : list
            List of probabilities associated with the above mixtures.
        """
        twotheta, intensity = self._load_pattern_from_file(filename)
        return super().predict(twotheta=twotheta, intensity=intensity)

    def _load_pattern_from_file(self, filename):
        """
        Loads a spectrum from a file.

        Parameters
        ----------
        filename : str
            Name of file to be analyzed.

        Returns
        -------
        twotheta : list
            Two-theta values of the spectrum.
        intensity : list
            Intensity values of the spectrum.
        """
        data = np.loadtxt(os.path.join(self.spectra_dir, filename))
        twotheta = data[:, 0]
        intensity = data[:, 1]

        return twotheta, intensity
