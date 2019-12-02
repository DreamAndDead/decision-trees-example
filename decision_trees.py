#!/usr/bin/env python3

"""
dataset: https://archive.ics.uci.edu/ml/datasets/Adult
"""

from graphviz import Digraph
import pandas as pd

ad = pd.read_csv('dataset/adult.data', sep=', ', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'target'], na_values='?', keep_default_na=False)


cat = ad.drop(columns=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']).dropna()


g = Digraph('g', filename='tree.gv')


class Node:
    def __init__(self, attr):
        self.attr = attr
        self.children = {}

    def add_child(self, attr, child):
        self.children[attr] = child

    def is_leaf(self):
        return len(self.children) == 0

attr_values = {}
for k in cat.keys():
    c = cat[k]
    attr_values[k] = c.unique()

    
def decision_tree(data):
    target = data['target']
    most_target = target.value_counts().keys()[0]
    
    # if all target is same
    if target.nunique() == 1:
        leaf = Leaf(target.unique()[0])
        return leaf

    # if attr is empty, return the most target
    if data.shape[1] == 1:
        leaf = Leaf(most_target)
        return leaf

    # pick attr
    attr_name = data.keys()[0]
    attr = data[attr_name]

    # split branch
    branch = Branch(attr_name)
    for v in attr_values[attr_name]:
        child_data = data[attr==v].drop(columns=[attr_name])
        if child_data.shape[0] == 0:
            leaf = Leaf(most_target)
            return leaf
        else:
            child = decision_tree(child_data)
            branch.add_child(v, child)

    return branch



t = decision_tree(cat)
