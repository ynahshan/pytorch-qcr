from .graph import Graph


def graph_to_model_namegraph(graph, model):
    module_name_map = dict([(m, n) for n, m in model.named_modules()])
    g = Graph()

    for v in graph.get_nodes():
        v_ = v.name if hasattr(v, 'name') else v
        if v.op_type == 'module':
            v_ += '_' + module_name_map[v.module]

        g.add_node(v_)

    for v1, v2 in graph.get_edges():
        v1_ = v1.name if hasattr(v1, 'name') else v1
        if v1.op_type == 'module':
            v1_ += '_' + module_name_map[v1.module]

        v2_ = v2.name if hasattr(v2, 'name') else v2
        if v2.op_type == 'module':
            v2_ += '_' + module_name_map[v2.module]

        g.add_edge(v1_, v2_)

    return g


class UndoInplace(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for m in self.model.modules():
            if hasattr(m, 'inplace'):
                m.inplace_modified = m.inplace
                m.inplace = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.model.modules():
            if hasattr(m, 'inplace_modified'):
                m.inplace = m.inplace_modified
                delattr(m, 'inplace_modified')
