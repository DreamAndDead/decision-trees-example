#!/usr/bin/env python3

"""
dataset: https://archive.ics.uci.edu/ml/datasets/Adult
"""

import pandas as pd

ad = pd.read_csv('dataset/adult.data', sep=', ', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'target'], na_values='?', keep_default_na=False)


cat = ad.drop(columns=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']).dropna()




class Branch:
    def __init__(self, attr):
        self.attr = attr
        self.children = {}

    def add_child(self, attr, child):
        self.children[attr] = child

class Leaf:
    def __init__(self, target):
        self.target = target



attr_values = {}
for k in cat.keys():
    c = cat[k]
    attr_values[k] = c.unique()

    
def decision_tree(data):
    target = data['target']
    most_target = target.value_counts().keys()[0]
    
    # if all target is same
    if target.nunique() == 1:
        return Leaf(target.unique()[0])

    # if attr is empty, return the most target
    if data.shape[1] == 1:
        return Leaf(most_target)

    # pick attr
    attr_name = data.keys()[0]
    attr = data[attr_name]

    # split branch
    branch = Branch(attr_name)
    for v in attr_values[attr_name]:
        child_data = data[attr==v].drop(columns=[attr_name])
        if child_data.shape[0] == 0:
            return Leaf(most_target)
        else:
            child = decision_tree(child_data)
            branch.add_child(v, child)

    return branch



t = decision_tree(cat)
