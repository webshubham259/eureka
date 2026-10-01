import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eureka import EUREKAClassifier, EUREKARegressor
from eureka.exceptions import DataDimensionError, NumericalInstabilityError


class EstimatorReliabilityTests(unittest.TestCase):
    def test_regressor_rejects_one_dimensional_features(self):
        model = EUREKARegressor()

        with self.assertRaisesRegex(DataDimensionError, "2D"):
            model.fit(np.array([1.0, 2.0]), np.array([2.0, 4.0]))

    def test_regressor_rejects_mismatched_target_length(self):
        model = EUREKARegressor()

        with self.assertRaisesRegex(DataDimensionError, "match X"):
            model.fit(np.ones((3, 1)), np.ones(2))

    def test_regressor_rejects_non_finite_features(self):
        model = EUREKARegressor()

        with self.assertRaises(NumericalInstabilityError):
            model.fit(np.array([[0.0], [np.nan]]), np.array([0.0, 1.0]))

    def test_classifier_rejects_single_class(self):
        model = EUREKAClassifier()

        with self.assertRaisesRegex(DataDimensionError, "at least two classes"):
            model.fit(np.ones((3, 1)), np.array(["only", "only", "only"]))

    def test_regression_fit_prediction_and_serialization(self):
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 1.0
        model = EUREKARegressor(alpha=0.001, top_k=1)
        model.fit(X, y)
        predictions = model.predict(X)

        with self.assertRaisesRegex(DataDimensionError, "expected 1"):
            model.predict(np.ones((2, 2)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regression-model"
            model.save(str(path))
            loaded_model = EUREKARegressor()
            loaded_model.load(str(path))

        np.testing.assert_allclose(loaded_model.predict(X), predictions)

    def test_neural_evaluation_does_not_prompt_for_input(self):
        X = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 1.0
        model = EUREKARegressor(
            alpha=0.001,
            augmentation_factor=2,
            sigma=0.01,
            epochs=1,
            batch_size=4,
            top_k=1,
            evaluate_nn=True,
        )

        with patch("builtins.input", side_effect=AssertionError("fit must not prompt")):
            model.fit(X, y)

        self.assertIsNotNone(model.symbolic_model)

    def test_classification_fit_prediction_and_serialization(self):
        feature = np.linspace(-1.0, 1.0, 20)
        X = np.column_stack([feature, feature ** 2])
        y = np.where(feature < 0.0, "left", "right")
        model = EUREKAClassifier(alpha=0.001, top_k=2)
        model.fit(X, y)
        predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classification-model"
            model.save(str(path))
            loaded_model = EUREKAClassifier()
            loaded_model.load(str(path))

        np.testing.assert_array_equal(loaded_model.predict(X), predictions)
        self.assertTrue(set(predictions).issubset({"left", "right"}))


if __name__ == "__main__":
    unittest.main()
