import graphviz
import networkx as nx
import pickle


def load(file):
    with open(file, 'rb') as handle:
        d = pickle.load(handle)
        return Graph(d)


class Graph(object):

    def __init__(self, gdict=None):
        if gdict is None:
            gdict = {}
        self.gdict = gdict

    def get_nodes(self):
        return list(self.gdict.keys())

    def get_edges(self):
        edgename = []
        for vrtx in self.gdict:
            for nxtvrtx in self.gdict[vrtx]:
                if {nxtvrtx, vrtx} not in edgename:
                    edgename.append((vrtx, nxtvrtx))
        return edgename

    def add_node(self, vrtx):
        if vrtx not in self.gdict:
            self.gdict[vrtx] = []

    def add_edge(self, vrtx1, vrtx2):
        if vrtx1 in self.gdict:
            self.gdict[vrtx1].append(vrtx2)
        else:
            self.gdict[vrtx1] = [vrtx2]

    def remove_edge(self, vrtx1, vrtx2):
        if vrtx1 in self.gdict:
            if vrtx2 in self.gdict[vrtx1]:
                self.gdict[vrtx1].remove(vrtx2)

    def get_node(self, node_name):
        for node in self.get_nodes():
            if node.name == node_name:
                return node

    def get_node_connections(self, node):
        return self.gdict[node]

    def get_nodes_by_predicate(self, predicate):
        res_nodes = []
        for node in self.get_nodes():
            if predicate(node):
                res_nodes.append(node)

        return res_nodes

    def to_namegraph(self):
        g = Graph()

        for v in self.get_nodes():
            g.add_node(v.name if hasattr(v, 'name') else v)

        for v1, v2 in self.get_edges():
            g.add_edge(v1.name if hasattr(v1, 'name') else v1, v2.name if hasattr(v2, 'name') else v2)

        return g

    def to_graphviz(self):
        g = graphviz.Digraph()

        for v in self.get_nodes():
            g.node(v.name) if hasattr(v, 'name') else v

        for v1, v2 in self.get_edges():
            g.edge(v1.name if hasattr(v1, 'name') else v1, v2.name if hasattr(v2, 'name') else v2)

        return g

    def to_nx(self):
        g = nx.DiGraph()

        for v in self.get_nodes():
            g.add_node(v.name) if hasattr(v, 'name') else v

        for v1, v2 in self.get_edges():
            g.add_edge(v1.name if hasattr(v1, 'name') else v1, v2.name if hasattr(v2, 'name') else v2)

        return g

    def save(self, file):
        with open(file, 'wb') as handle:
            pickle.dump(self.gdict, handle)

    def __repr__(self):
        return str(self.gdict)
