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

    def test_view(self):
        self.tree.view_graph()

    def test_error_rate(self):
        sum = 0
        error = 0
        for _, row in self.test.iterrows():
            sum += 1
            
            p = dt.predict(self.tree, row)
            t = row['target']

            if p != t:
                error += 1
                print(row.to_string())
                print("mispredict target to %s" % p)
                print()
                
        print("error rate = %d/%d = %f" % (error, sum, error / sum))
        
