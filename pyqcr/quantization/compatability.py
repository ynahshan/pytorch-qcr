import torch


def is_substring(s, sub_list):
    return len([sub for sub in sub_list if sub in s]) > 0


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


def check_quantization_compatability(model):
    found_issues = False
    reused = get_reused_modules(model)
    if len(reused['model_names']) > 0:
        found_issues = True
        print(
            "Found {} reused modules. Those modules has to be converted to unique instanses. Run compatability.get_reused_modules(model) to get the list.\n".format(
                len(reused['model_names'])))

    non_module_ops = get_non_module_ops(model)
    ff_ops = [op for op in non_module_ops if
              is_substring(op, ['add', 'cat', 'mul', 'add_relu', 'add_scalar', 'mul_scalar'])]
    if len(ff_ops) > 0:
        found_issues = True
        print(
            "Found {} non module operations that should be converted to nn.quantized.FloatFunctional. compatability comp.get_non_module_ops(model) for more information.\n".format(
                len(ff_ops)))

    inplace_modules = [m for m in model.modules() if hasattr(m, 'inplace') and m.inplace == True]
    if len(inplace_modules) > 0:
        found_issues = True
        print("Found {} inplace modules. In order to quantize this model set inplace=Fales instead.\n".format(
            len(inplace_modules)))

    if not found_issues:
        print("No issues found. Model should be compatible with quantization.")
