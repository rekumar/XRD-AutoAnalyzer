import tensorflow as tf


# Used to apply dropout during training *and* inference
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class KerasDropoutPrediction(object):
    """
    Ensemble model used to provide a probability distribution associated
    with suspected phases in a given xrd spectrum.
    """

    def __init__(self, model):
        """
        Args:
            model: trained convolutional neural network
                (tensorflow.keras Model object)
        """

        self.model = model

    def predict(self, x, min_conf=10.0, n_iter=100):
        """
        Args:
            x: xrd spectrum to be classified
        Returns:
            prediction: distribution of probabilities associated with reference phases
            len(certainties): number of phases with probabilities > 10%
            certanties: associated probabilities
        """

        # Convert from % to 0-1 fractional
        if min_conf > 1.0:
            min_conf /= 100.0

        # Format input
        x = [[val] for val in x]
        x = np.array([x])

        # Monte Carlo Dropout
        result = []
        for _ in range(n_iter):
            result.append(self.model(x))

        result = np.array(
            [list(np.array(sublist).flatten()) for sublist in result]
        )  ## Individual predictions
        prediction = result.mean(axis=0)  ## Average prediction

        all_preds = [
            np.argmax(pred) for pred in result
        ]  ## Individual max indices (associated with phases)

        counts = []
        for index in set(all_preds):
            counts.append(
                all_preds.count(index)
            )  ## Tabulate how many times each prediction arises

        certanties = []
        for each_count in counts:
            conf = each_count / sum(counts)
            if conf >= min_conf:
                certanties.append(conf)
        certanties = sorted(certanties, reverse=True)

        return prediction, len(certanties), certanties
