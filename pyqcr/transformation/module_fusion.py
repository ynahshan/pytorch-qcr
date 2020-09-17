import copy
import torch
import torch.nn as nn


DEFAULT_PATTERNS = [
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU),
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU),
    (nn.Conv1d, nn.BatchNorm1d),
    (nn.Conv2d, nn.BatchNorm2d),
    (nn.Conv3d, nn.BatchNorm3d),
    (nn.BatchNorm2d, nn.ReLU),
    (nn.BatchNorm3d, nn.ReLU),
    (nn.Conv1d, nn.ReLU),
    (nn.Conv2d, nn.ReLU),
    (nn.Conv3d, nn.ReLU),
    (nn.Linear, nn.ReLU),
]


class Fuser(object):
    def __init__(self):
        pass

    def _check_match(self, graph, node, pattern):
        res_modules = []
        if node.op_type == 'module' and isinstance(node.module, pattern[0]):
            res_modules.append(node)
            for i in range(1, len(pattern)):
                conn_nodes = graph.get_node_connections(node)
                if len(conn_nodes) == 1 and conn_nodes[0].op_type == 'module' and isinstance(conn_nodes[0].module,
                                                                                             pattern[i]):
                    node = conn_nodes[0]
                    res_modules.append(node)

        return res_modules if len(res_modules) == len(pattern) else []

    def get_default_patterns(self):
        return DEFAULT_PATTERNS

    def find_matches(self, graph, pattern):
        matches = []
        for node in graph.get_nodes():
            match = self._check_match(graph, node, pattern)
            if len(match) > 0:
                matches.append(match)

        return matches

    def find_fusable_modules(self, model, patterns=DEFAULT_PATTERNS):
        graph = model.graph

        fused_modules = set()
        pattern_fuse_map = []
        module_name_map = dict([(m, n) for n, m in model.named_modules()])
        for pattern in patterns:
            matches = self.find_matches(graph, pattern)
            if len(matches) > 0:
                modules_to_fuse = [[module_name_map[node.module] for node in match] for match in matches]
                new_fuses = [mf for mf in modules_to_fuse if not any(m in fused_modules for m in mf)]
                for mf in new_fuses:
                    fused_modules.update(mf)

                if len(new_fuses) > 0:
                    pattern_fuse_map.append((pattern, new_fuses))

        return pattern_fuse_map

    def fuse(self, model, inplace=False, patterns=DEFAULT_PATTERNS):
        if not hasattr(model, 'graph') or model.graph == None:
            raise RuntimeError("Model does not have graph attribute.")

        fusable = self.find_fusable_modules(model, patterns)
        modules_to_fuse = []
        for p, modules in fusable:
            modules_to_fuse += modules

        model = torch.quantization.fuse_modules(model, modules_to_fuse=modules_to_fuse, inplace=inplace)
        # TODO: modify graph

        return model
