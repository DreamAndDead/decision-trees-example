from graphviz import Digraph
import math

class Node:
    count = 0
    graph = Digraph('tree', filename='tree.gv')
    
    def __init__(self, value):
        self.number = str(Node.count)
        Node.count += 1
        
        self.value = value
        Node.graph.node(self.number, self.value)
        self.children = {}

    def add_child(self, value, child):
        self.children[value] = child
        Node.graph.edge(self.number, child.number, label=value)

    def is_leaf(self):
        return len(self.children) == 0

    def reset_graph(self):
        Node.graph.clear()

    def view_graph(self):
        Node.graph.render(view=True, format='png', renderer='cairo', formatter='cairo')


def entropy(data):
    target = data['target']
    sum = len(target)
    count = target.value_counts()

    entro = 0
    for v in count.values:
        p = v / sum
        if p != 0:
            entro += (-p * math.log2(p))

    return entro


def pick_feature(data):
    entro_before = entropy(data)

    entro_gain = -math.inf
    feature_name = None
    feature_names = data.columns.drop('target')

    for f in feature_names:
        col = data[f]
        sum = len(col)
        count = col.value_counts()
        
        entro_after = 0
        for k in count.keys():
            p = count[k] / sum
            entro_after += p * entropy(data[col==k])

        gain = entro_before - entro_after
        if gain > entro_gain:
            entro_gain = gain
            feature_name = f

    return feature_name

    
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
        if data.shape[1] == 1:   # only target column
            node = Node(most_target)
            return node

        feature_name = pick_feature(data)
        feature = data[feature_name]

        node = Node(feature_name)
        for v in feature_values[feature_name]:
            child_data = data[feature==v].drop(columns=[feature_name])

            if child_data.shape[0] == 0:
                child = Node(most_target)
            else:
                child = _decision_tree(child_data)

            node.add_child(v, child)

        return node

    tree = _decision_tree(data)
    return tree


def predict(tree, data):
    if tree.is_leaf():
        return tree.value

    feature_name = tree.value
    feature_value = data[feature_name]
    return predict(tree.children[feature_value], data)

