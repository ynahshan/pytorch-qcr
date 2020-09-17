import torch


def get_reused_modules(model):
    if not hasattr(model, 'graph') or model.graph == None:
        raise RuntimeError("Model does not have graph attribute.")

    module_name_map = dict([(m, n) for n, m in model.named_modules()])
    used_modules = set()
    reused_modules_graph_names = []
    reused_modules_model_names = []
    module_nodes = [node for node in model.graph.get_nodes() if node.op_type == 'module']

    for node in module_nodes:
        if node.module in used_modules:
            # module already was used in the graph
            reused_modules_graph_names.append(node.name)
            reused_modules_model_names.append(module_name_map[node.module])
        else:
            used_modules.add(node.module)

    return {'graph_names': reused_modules_graph_names, 'model_names': reused_modules_model_names}


def get_non_module_ops(model):
    if not hasattr(model, 'graph') or model.graph == None:
        raise RuntimeError("Model does not have graph attribute.")

    func_nodes = [node.name for node in model.graph.get_nodes() if node.op_type == 'function']
    return func_nodes
