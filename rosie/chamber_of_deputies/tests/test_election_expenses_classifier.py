from unittest import TestCase

import numpy as np
import pandas as pd

from rosie.chamber_of_deputies.classifiers import ElectionExpensesClassifier


class TestElectionExpensesClassifier(TestCase):

    def setUp(self):
        self.dataset = pd.read_csv('rosie/chamber_of_deputies/tests/fixtures/election_expenses_classifier.csv',
                                   dtype={'name': np.str, 'legal_entity': np.str})
        self.subject = ElectionExpensesClassifier()

    def test_is_election_company(self):
        self.assertEqual(self.subject.predict(self.dataset)[0], True)

    def test_is_not_election_company(self):
        self.assertEqual(self.subject.predict(self.dataset)[1], False)

    def test_fit_just_for_formality_because_its_never_used(self):
        self.assertTrue(self.subject.fit(self.dataset) is None)

    def test_transform_just_for_formality_because_its_never_used(self):
        self.assertTrue(self.subject.transform() is None)
