import unittest
import pandas as pd
import decision_trees as dt

class TestDecisionTrees(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_excel('triple_all.xlsx')
        self.tree = dt.decision_tree(self.data)

    def tearDown(self):
        self.tree.reset_graph()
    
    def test_origin(self):
        for _, row in self.data.iterrows():
            res = dt.predict(self.tree, row)
            self.assertEqual(res, row['target'])

        self.tree.view_graph()

