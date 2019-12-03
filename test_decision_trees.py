"""
dataset: https://archive.ics.uci.edu/ml/datasets/Adult
"""

import unittest
import pandas as pd
import decision_trees as dt

class TestDecisionTrees(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('dataset/adult.data', sep=', ', na_values='?', keep_default_na=False, engine='python')

        self.cat_data = self.data.drop(columns=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']).dropna()

        self.data = self.cat_data.iloc[:100]

        self.tree = dt.decision_tree(self.data)

    def tearDown(self):
        self.tree.reset_graph()
    
    def test_origin(self):
        for _, row in self.data.iterrows():
            res = dt.predict(self.tree, row)
            self.assertEqual(res, row['target'])

        self.tree.view_graph()

