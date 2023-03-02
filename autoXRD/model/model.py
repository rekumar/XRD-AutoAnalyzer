import os
import numpy as np
import tensorflow as tf
from random import shuffle
from autoXRD.cnn.dropout import CustomDropout
from warnings import warn
import h5py
from typing import Dict, List, Optional, Tuple
from pymatgen.core import Structure


class TrainingDataHandler(object):
    """
    Class used to train a convolutional neural network on a given
    set of X-ray diffraction spectra to perform phase identification.
    """

    def __init__(self, xrd, testing_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by
                their associated reference phase.
                The shape of the array should be NxMx4501x1 where:
                N = the number of reference phases,
                M = the number of augmented spectra per reference phase,
                4501 = intensities as a function of 2-theta
                (spanning from 10 to 80 degrees by default)
            testing_fraction: fraction of data (xrd patterns) to reserve for testing.
                By default, all spectra will be used for training.
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)

    @property
    def phase_indices(self):
        """
        List of indices to keep track of xrd spectra such that
            each index is associated with a reference phase.
        """
        return [v for v in range(self.num_phases)]

    @property
    def x(self):
        """
        Feature matrix (array of intensities used for training)
        """
        intensities = []
        xrd = self.xrd
        phase_indices = self.phase_indices
        for (augmented_spectra, index) in zip(xrd, phase_indices):
            for pattern in augmented_spectra:
                intensities.append(pattern)
        return np.array(intensities)

    @property
    def y(self):
        """
        Target property to predict (one-hot encoded vectors associated
        with the reference phases)
        """
        xrd = self.xrd
        phase_indices = self.phase_indices
        one_hot_vectors = []
        for (augmented_spectra, index) in zip(xrd, phase_indices):
            for pattern in augmented_spectra:
                assigned_vec = [[0]] * len(xrd)
                assigned_vec[index] = [1.0]
                one_hot_vectors.append(assigned_vec)
        return np.array(one_hot_vectors)

    def split_training_testing(self):
        """
        Training and testing data will be split according
        to self.testing_fraction

        Returns:
            x_train, x_test: features matrices (xrd spectra) to be
                used for training and testing
            y_train, t_test: target properties (one-hot encoded phase indices)
                to be used for training and testing
        """
        x = self.x
        y = self.y
        testing_fraction = self.testing_fraction
        combined_xy = list(zip(x, y))
        shuffle(combined_xy)

        if testing_fraction == 0:
            train_x, train_y = zip(*combined_xy)
            test_x, test_y = None, None
            return np.array(train_x), np.array(train_y), test_x, test_y

        else:
            total_samples = len(combined_xy)
            n_testing = int(testing_fraction * total_samples)

            train_xy = combined_xy[n_testing:]
            train_x, train_y = zip(*train_xy)

            test_xy = combined_xy[:n_testing]
            test_x, test_y = zip(*test_xy)

            return (
                np.array(train_x),
                np.array(train_y),
                np.array(test_x),
                np.array(test_y),
            )


def train_model(
    x_train,
    y_train,
    n_phases,
    num_epochs,
    is_pdf,
    n_dense=[3100, 1200],
    dropout_rate=0.7,
) -> tf.keras.Model:
    """
    Args:
        x_train: numpy array of simulated xrd spectra
        y_train: one-hot encoded vectors associated with reference phase indices
        n_phases: number of reference phases considered
        fmodel: filename to save trained model to
        n_dense: number of nodes comprising the two hidden layers in the neural network
        dropout_rate: fraction of connections excluded between the hidden layers during training
    Returns:
        model: trained and compiled tensorflow.keras.Model object
    """

    # Optimized architecture for PDF analysis
    if is_pdf:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=60,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.Flatten(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_dense[0], activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_dense[1], activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_phases, activation=tf.nn.softmax),
            ]
        )

    # Optimized architecture for XRD analysis
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=35,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same"),
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=30,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same"),
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=25,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=20,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=15,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=10,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same"),
                tf.keras.layers.Flatten(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_dense[0], activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_dense[1], activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                CustomDropout(dropout_rate),
                tf.keras.layers.Dense(n_phases, activation=tf.nn.softmax),
            ]
        )

    # Compile model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Fit model to training data
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=num_epochs,
        validation_split=0.2,
        shuffle=True,
    )

    return model


def test_model(model, test_x, test_y):
    """
    Args:
        model: trained tensorflow.keras.Model object
        x_test: feature matrix containing xrd spectra
        y_test: one-hot encoded vectors associated with
            the reference phases
    """
    _, acc = model.evaluate(test_x, test_y)
    print("Test Accuracy: " + str(round(acc * 100, 2)) + "%")
    return acc


def main(
    xrd: np.ndarray,
    reference_structures: Dict[str, Structure],
    num_epochs: int,
    testing_fraction: float,
    is_pdf: bool,
    savepath: Optional[str] = None,
):
    if len(reference_structures) != xrd.shape[0]:
        raise ValueError(
            "Length of reference_names must match number of rows in xrd matrix!"
        )
    if savepath is None:
        if is_pdf:
            savepath = "Model_PDF.autoxrd.h5"
        else:
            savepath = "Model_XRD.autoxrd.h5"
    elif not savepath.endswith(".h5"):
        raise ValueError("Model file must be a .h5 file. You provided: " + savepath)

    # Organize data
    data_handler = TrainingDataHandler(xrd, testing_fraction)
    num_phases = data_handler.num_phases
    train_x, train_y, test_x, test_y = data_handler.split_training_testing()

    # Train model
    model = train_model(train_x, train_y, num_phases, num_epochs, is_pdf)

    # Save model
    model.save(savepath, include_optimizer=False)

    # save reference structures inside the model h5 file
    #
    # file
    #   - model (tensorflow)
    #   - reference_structures (same order as model)
    #       - name (list of strings)
    #       - cif (list of strings to be loaded with Structure.from_str(s, fmt="cif"))
    with h5py.File(savepath, "a") as f:
        g = f.create_group(name="reference_structures")
        g.create_dataset(name="name", data=list(reference_structures.keys()))
        g.create_dataset(
            name="cif", data=[s.to(fmt="cif") for s in reference_structures.values()]
        )

    # Test model is any data is reserved for testing
    if testing_fraction != 0:
        test_model(model, test_x, test_y)


def load_model(model_path: str) -> Tuple[tf.keras.Model, Dict[str, Structure]]:
    """
    Args:
        model_path: path to model file
    Returns:
        model: tensorflow.keras.Model object
        reference structures: Dict of [reference name: pymatgen Structure of reference]
    """
    model = tf.keras.models.load_model(model_path)
    with h5py.File(model_path, "r") as f:
        reference_names = list(f["reference_structures"]["name"])
        reference_structures = [
            Structure.from_str(s, fmt="cif") for s in f["reference_structures"]["cif"]
        ]
    return model, {
        name: structure
        for name, structure in zip(reference_names, reference_structures)
    }
