# Copyright (c) 2024-2025 Lucas Leão
# tinyCP - A small toolbox for conformal prediction
# Licensed under the MIT License


import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tinyconformal.classifier.marginal import BinaryMarginalConformalClassifier
from tinyconformal.classifier.class_conditional import (
    BinaryClassConditionalConformalClassifier,
)


class TestClassifiers(unittest.TestCase):
    def setUp(self):
        weights = [0.4, 0.6]
        seed = 42

        X, y = make_classification(
            n_samples=1500,
            n_features=20,
            n_informative=2,
            weights=weights,
            random_state=seed,
            n_redundant=2,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
        )

        self.X_train = X_train
        self.y_train = y_train

        self.X_calib = X_calib
        self.y_calib = y_calib

        self.X_test = X_test
        self.y_test = y_test

        self.learner = RandomForestClassifier(oob_score=True, n_estimators=10)
        self.learner.fit(self.X_train, self.y_train)

    def test_marginal_classifier(self):
        classifier = BinaryMarginalConformalClassifier(self.learner)
        classifier.fit(self.X_calib, self.y_calib, oob=False)

        classifier.calibrate(self.X_calib, self.y_calib)
        self.assertTrue(0 < classifier.alpha <= 0.2)

        y_proba = classifier.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (self.X_test.shape[0], 2))

        prediction_set = classifier.predict_set(self.X_test)
        self.assertEqual(prediction_set.shape, (self.X_test.shape[0], 2))

        p_values = classifier.predict_p(self.X_test)
        self.assertEqual(p_values.shape, (self.X_test.shape[0], 2))

        y_pred = classifier.predict(self.X_test)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))

        eval_dict = classifier.evaluate(self.X_test, self.y_test)
        self.assertTrue(isinstance(eval_dict, dict))
        self.assertEqual(len(eval_dict.keys()), 13)

    def test_class_cond_classifier(self):
        classifier = BinaryClassConditionalConformalClassifier(self.learner)
        classifier.fit(self.X_calib, self.y_calib, oob=False)

        classifier.calibrate(self.X_calib, self.y_calib)
        self.assertTrue(0 < classifier.alpha <= 0.2)

        y_proba = classifier.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (self.X_test.shape[0], 2))

        prediction_set = classifier.predict_set(self.X_test)
        self.assertEqual(prediction_set.shape, (self.X_test.shape[0], 2))

        p_values = classifier.predict_p(self.X_test)
        self.assertEqual(p_values.shape, (self.X_test.shape[0], 2))

        y_pred = classifier.predict(self.X_test)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))

        eval_dict = classifier.evaluate(self.X_test, self.y_test)
        self.assertTrue(isinstance(eval_dict, dict))
        self.assertEqual(len(eval_dict.keys()), 13)

    def test_oob_marginal_classifier(self):
        classifier = BinaryMarginalConformalClassifier(self.learner)
        classifier.fit(y=self.y_train, oob=True)

        classifier.calibrate(self.X_calib, self.y_calib)
        self.assertTrue(0 < classifier.alpha <= 0.2)

        y_proba = classifier.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (self.X_test.shape[0], 2))

        prediction_set = classifier.predict_set(self.X_test)
        self.assertEqual(prediction_set.shape, (self.X_test.shape[0], 2))

        p_values = classifier.predict_p(self.X_test)
        self.assertEqual(p_values.shape, (self.X_test.shape[0], 2))

        y_pred = classifier.predict(self.X_test)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))

        eval_dict = classifier.evaluate(self.X_test, self.y_test)
        self.assertTrue(isinstance(eval_dict, dict))
        self.assertEqual(len(eval_dict.keys()), 13)

    def test_oob_class_conditional_classifier(self):
        classifier = BinaryClassConditionalConformalClassifier(self.learner)
        classifier.fit(y=self.y_train, oob=True)

        classifier.calibrate(self.X_calib, self.y_calib)
        self.assertTrue(0 < classifier.alpha <= 0.2)

        y_proba = classifier.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (self.X_test.shape[0], 2))

        prediction_set = classifier.predict_set(self.X_test)
        self.assertEqual(prediction_set.shape, (self.X_test.shape[0], 2))

        p_values = classifier.predict_p(self.X_test)
        self.assertEqual(p_values.shape, (self.X_test.shape[0], 2))

        y_pred = classifier.predict(self.X_test)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))

        eval_dict = classifier.evaluate(self.X_test, self.y_test)
        self.assertTrue(isinstance(eval_dict, dict))
        self.assertEqual(len(eval_dict.keys()), 13)


if __name__ == "__main__":
    unittest.main()
