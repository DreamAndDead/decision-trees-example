#!/usr/bin/env python3

from graphviz import Digraph
import math

class Node:
    count = 0
    graph = Digraph('tree', filename='tree.gv')
    
    def __init__(self, feature):
        self.number = str(Node.count)
        Node.count += 1
        
        self.feature = feature
        self.children = {}
        Node.graph.node(self.number, feature)

    def add_child(self, value, child):
        self.children[value] = child
        Node.graph.edge(self.number, child.number, label=value)

    def is_leaf(self):
        return len(self.children) == 0

    def reset_graph(self):
        Node.graph.clear()

    def view_graph(self):
        Node.graph.view()

def entropy(data):
    """
    equals sum -p*log2(p)
    """
    target = data['target']
    count = target.value_counts()
    sum = len(target)

    entro = 0
    for k in count.keys():
        p = count[k] / sum
        if p == 0:
            entro += 0
        else:
            entro += (-p * math.log2(p))

    return entro

def pick_feature_by_entropy(data):
    """return feature name"""
    origin_entro = entropy(data)

    entro_gain = 0
    feature = None
    
    features = data.columns.drop('target')
    for f in features:
        col = data[f]
        sum = len(col)

        count = col.value_counts()
        new_entro = 0
        for k in count.keys():
            p = count[k] / sum
            new_entro += p * entropy(data[col==k])

        gain = origin_entro - new_entro
        if gain > entro_gain:
            entro_gain = gain
            feature = f

    return feature
    
def decision_tree(data):
    feature_values = {}
    for k in data.keys():
        feature_values[k] = data[k].unique()

    def _decision_tree(data):
        target = data['target']
        most_target = target.value_counts().keys()[0]
        
        # if all target is same
        if target.nunique() == 1:
            node = Node(target.unique()[0])
            return node

        # if feature is empty, return the most target
        if data.shape[1] == 1:
            node = Node(most_target)
            return node

        # pick feature
        feature_name = pick_feature_by_entropy(data)
        feature = data[feature_name]

        # split node
        node = Node(feature_name)
        for v in feature_values[feature_name]:
            child_data = data[feature==v].drop(columns=[feature_name])
            if child_data.shape[0] == 0:
                child = Node(most_target)
                node.add_child(v, child)
            else:
                child = _decision_tree(child_data)
                node.add_child(v, child)

        return node

    tree = _decision_tree(data)
    return tree


def predict(tree, data):
    if tree.is_leaf():
        return tree.feature

    feature = tree.feature
    feature_value = data[feature]
    return predict(tree.children[feature_value], data)

