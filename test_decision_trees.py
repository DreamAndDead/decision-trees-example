import unittest
import pandas as pd
import decision_trees as dt

class TestDecisionTrees(unittest.TestCase):
    def setUp(self):
        self.train = pd.read_excel('dataset/offence_train.xlsx')
        self.test = pd.read_excel('dataset/offence_test.xlsx')
        self.tree = dt.decision_tree(self.train)

    def tearDown(self):
        self.tree.reset_graph()
    
    def test_origin(self):
        for _, row in self.test.iterrows():
            res = dt.predict(self.tree, row)
            self.assertEqual(res, row['target'])

        self.tree.view_graph()

