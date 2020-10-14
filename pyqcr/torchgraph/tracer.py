import torch
import inspect
import types
from functools import wraps
import networkx as nx
from .graph import Graph
from .utils import UndoInplace


class OpTensor(object):
    def __init__(self, inp, name=''):
        self.name = name
        self.type = type(inp)
        self._istensor = isinstance(inp, torch.Tensor)
        if self._istensor:
            self.shape = inp.shape
            self.dtype = inp.dtype
            self.id = inp.data_ptr()
            self.tensor_id = id(inp)
            self.data = inp.sum().item()

    @property
    def is_tensor(self):
        return self._istensor

    def __repr__(self):
        res = "name: {}, type: {}".format(self.name, self.type)
        if self._istensor:
            tensor_type = self.type.__name__
            res1 = ", {}: shape {}, dtype {}, id {}, tensor_id {}, data {}".format(tensor_type, self.shape, self.dtype, self.id, self.tensor_id, self.data)
            res += res1
        return res


class TorchOp(object):
    def __init__(self, inputs=[], outputs=[]):
        self.inputs = inputs
        self.outputs = outputs
        self._isinplace = False
        inp_ids = [inp.id for inp in self.inputs if inp.is_tensor]
        for out in self.outputs:
            if out.is_tensor and out.id in inp_ids:
                self._isinplace = True

    @staticmethod
    def parse_args(args=[]):
        op_args = []
        if isinstance(args, list) or isinstance(args, tuple):
            for arg in args:
                if isinstance(arg, list) or isinstance(arg, tuple):
                    for inp in arg:
                        op_args.append(OpTensor(inp))
                else:
                    op_args.append(OpTensor(arg))
        else:
            op_args.append(OpTensor(args))

        return op_args

    @property
    def is_inplace(self):
        return self._isinplace

    @property
    def op_name(self):
        raise NotImplementedError

    def __repr__(self):
        res = "({} - {}, \ninputs:\n".format(self.__class__.__name__, self.op_name)
        for i, inp in enumerate(self.inputs):
            res += '\ninput{} - '.format(i) + inp.__repr__()
        res += "\n outputs: "
        for i, inp in enumerate(self.outputs):
            res += '\noutput{} - '.format(i) + inp.__repr__()
        res += "\n)"
        return res


class FuncOp(TorchOp):
    def __init__(self, func, inputs, outputs):
        super(FuncOp, self).__init__(inputs, outputs)
        self._op_name = func.__name__
        self.op_type = 'function'

    @property
    def op_name(self):
        return self._op_name


class ModuleOp(TorchOp):
    def __init__(self, module, inputs, outputs):
        super(ModuleOp, self).__init__(inputs, outputs)
        self.module = module
        self.ops = []
        self.op_type = 'module'

    @property
    def op_name(self):
        return self.module.__class__.__name__


class ScalarOp(TorchOp):
    def __init__(self, name, inp_id=None, out_id=None):
        super(ScalarOp, self).__init__()
        self._op_name = name
        self.name = name
        self.inp_id = inp_id
        self.out_id = out_id
        self.op_type = 'scalar'

    @property
    def op_name(self):
        return self._op_name

    def __repr__(self):
        res = super(ScalarOp, self).__repr__()
        if self.inp_id is not None:
            res += "\ninput id: {}".format(self.inp_id)
        if self.out_id is not None:
            res += "\noutput id: {}".format(self.out_id)
        return res


class InputOp(TorchOp):
    def __init__(self, name, out_id=None):
        super(InputOp, self).__init__()
        self._op_name = name
        self.name = name
        self.out_id = out_id
        self.op_type = 'input'

    @property
    def op_name(self):
        return self._op_name

    def __repr__(self):
        res = super(InputOp, self).__repr__()
        if self.out_id is not None:
            res += "\noutput id: {}".format(self.out_id)
        return res


class OutputOp(TorchOp):
    def __init__(self, name, inp_id=None):
        super(OutputOp, self).__init__()
        self._op_name = name
        self.name = name
        self.inp_id = inp_id
        self.op_type = 'output'

    @property
    def op_name(self):
        return self._op_name

    def __repr__(self):
        res = super(OutputOp, self).__repr__()
        if self.inp_id is not None:
            res += "\ninput id: {}".format(self.inp_id)
        return res


class TorchTracer(object):
    def __init__(self):
        self._trace = []
        self._main_trace = None
        self._tracing_enabled = True

    def __enter__(self):
        self.trace_ = []
        self.torch = types.SimpleNamespace()
        self.Tensor = types.SimpleNamespace()
        self.nn_functional = types.SimpleNamespace()
        setattr(self.torch, 'funcs', [])
        setattr(self.Tensor, 'funcs', [])
        setattr(self.nn_functional, 'funcs', [])

        # Wrap torch methods
        for name in dir(torch._C._VariableFunctions):
            if name.startswith('__') or name.startswith('_'):
                continue
            if hasattr(torch, name):
                #             print(func.__name__)
                func = getattr(torch, name)
                self.torch.funcs.append(name)
                setattr(self.torch, name, func)
                setattr(torch, name, self.wrap_func(func))

        # Wrap torch.Tensor methods
        tensor_methods = self._get_tensor_methods()
        for name, func in tensor_methods:
            if hasattr(torch.Tensor, name):
                self.Tensor.funcs.append(name)
                setattr(self.Tensor, name, func)
                setattr(torch.Tensor, name, self.wrap_func(func))

        # Wrap torch.nn.functional methods
        nn_methods = self._get_nn_functional_methods()
        for name, func in nn_methods:
            if hasattr(torch.nn.functional, name):
                self.nn_functional.funcs.append(name)
                setattr(self.nn_functional, name, func)
                setattr(torch.nn.functional, name, self.wrap_func(func))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self.torch.funcs:
            setattr(torch, name, getattr(self.torch, name))
        for name in self.Tensor.funcs:
            setattr(torch.Tensor, name, getattr(self.Tensor, name))
        for name in self.nn_functional.funcs:
            setattr(torch.nn.functional, name, getattr(self.nn_functional, name))

    class NoTrace(object):
        def __init__(self, tracer):
            self.tracer = tracer

        def __enter__(self):
            self.tracer.__exit__(None, None, None)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracer.__enter__()

    def no_trace(self):
        return TorchTracer.NoTrace(self)

    def _get_tensor_methods(self):
        exclude_methods = ['__format__', '__dir__', '__len__', '__sizeof__', '__bool__', '__float__', '__hash__',
                           '__int__', '_is_view', '_make_subclass', '_values', 'data_ptr', 'type',
                           'type_as', 'detach', 'dim', 'flatten', 'numel', 'size', 'to']

        wrapper_descriptor = type(torch.Tensor.__getattribute__)
        all_methods = inspect.getmembers(torch.Tensor, predicate=inspect.isroutine)
        tensor_methods = [f for f in all_methods if type(f[1]) != wrapper_descriptor and f[0] not in exclude_methods]
        return tensor_methods

    def _get_nn_functional_methods(self):
        all_methods = [f for f in inspect.getmembers(torch.nn.functional, predicate=inspect.isroutine) if
                       not f[0].startswith('_')]
        exclude_torch = [f for f in all_methods if not hasattr(torch, f[0])]
        exclude_tensor = [f for f in exclude_torch if not hasattr(torch.Tensor, f[0])]
        return exclude_tensor

    def wrap_func(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            #             print(func.__qualname__)
            with self.no_trace():
                inputs = FuncOp.parse_args(args)
                result = func(*args, **kwargs)
                outputs = FuncOp.parse_args(result)
                op = FuncOp(func, inputs, outputs)
                self.trace.append(op)

            return result

        return wrapper

    @property
    def trace(self):
        return self._trace

    def redirect_trace(self, op_list):
        self._main_trace = self._trace
        self._trace = op_list

    def restore_tracing(self):
        if self._main_trace is not None:
            self._trace = self._main_trace
            self._main_trace = None

    def trace_model(self, model, *args, **kwargs):
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise RuntimeError("TorchTracer does not support DataParallel and DistributedDataParallel")

        module_ops = {}

        def pre_hook(module, input):
            module_ops[module] = []
            self.redirect_trace(module_ops[module])
            with self.no_trace():
                module._inputs = ModuleOp.parse_args(input)

        def hook(module, input, output):
            self.restore_tracing()
            with self.no_trace():
                outputs = ModuleOp.parse_args(output)
                mop = ModuleOp(module, module._inputs, outputs)
                delattr(module, '_inputs')
            mop.ops = module_ops[module]
            self.trace.append(mop)

        leafs = [m for m in model.modules() if len([m for m in m.children()]) == 0]
        #     print(leafs)

        handles = []
        for m in leafs:
            handles.append(m.register_forward_pre_hook(pre_hook))
            handles.append(m.register_forward_hook(hook))

        # with torch.no_grad():
        with UndoInplace(model):
            model(*args, **kwargs)

        for h in handles:
            h.remove()

    def to_graph(self):
        return TorchTracer.trace_to_graph(self.trace)

    def node_to_graph(self, node):
        if isinstance(node, ModuleOp):
            return TorchTracer.trace_to_graph(node.ops)
        elif isinstance(node, FuncOp):
            return TorchTracer.trace_to_graph([node])
        elif isinstance(node, ScalarOp):
            g = Graph()
            g.add_node(node)
            return g
        else:
            raise ValueError("node is invalid")

    @staticmethod
    def trace_to_graph(trace):
        # Create unique names for ops
        op_counter = {}
        for op in trace:
            if op.op_name in op_counter:
                op_counter[op.op_name] += 1
            else:
                op_counter[op.op_name] = 0

            op.name = "{}{}".format(op.op_name, op_counter[op.op_name])

        tensor_counter = const_counter = scalar_counter = 0
        # Find all connections
        conn = {}
        for i, op in enumerate(trace):
            op.idx = i
            for inp in op.inputs:
                if inp.is_tensor:
                    if inp.id not in conn:
                        conn[inp.id] = type('', (object,), {"consumers": [], "producers": []})()
                        conn[inp.id].type = inp.type

                    if op not in conn[inp.id].consumers:
                        conn[inp.id].consumers.append(op)
                else:
                    conn[id(inp)] = type('', (object,),
                                         {"consumers": [op], "producers": [ScalarOp('const{}'.format(const_counter), out_id=id(inp))]})()
                    const_counter += 1
            for out in op.outputs:
                if out.is_tensor:
                    if out.id not in conn:
                        conn[out.id] = type('', (object,), {"consumers": [], "producers": []})()
                        conn[out.id].type = out.type

                    if op not in conn[out.id].producers:
                        conn[out.id].producers.append(op)
                else:
                    conn[id(out)] = type('', (object,),
                                               {"consumers": [ScalarOp('scalar{}'.format(scalar_counter), inp_id=id(out))], "producers": [op]})()
                    scalar_counter += 1

        # create input/output nodes for not connected tensors (inputs/ouputs)
        for e in conn:
            if len(conn[e].consumers) == 0:
                op = OutputOp('{}{}'.format(conn[e].type.__name__, tensor_counter), inp_id=e)
                trace.append(op)
                conn[e].consumers.append(op)
                tensor_counter += 1
            if len(conn[e].producers) == 0 or (len(conn[e].producers) == 1 and conn[e].producers[0].is_inplace):
                op = InputOp('{}{}'.format(str(conn[e].type.__name__), tensor_counter), out_id=e)
                trace.insert(0, op)
                conn[e].producers.append(op)
                tensor_counter += 1

        def is_filtered(op):
            filter = [ScalarOp]
            for op_type in filter:
                if isinstance(op, op_type):
                    return True

            return False

        # create graph from connections
        g = Graph()

        # add ops as nodes
        for tid in conn:
            for c in conn[tid].consumers:
                if not is_filtered(c):
                    g.add_node(c)
            for p in conn[tid].producers:
                if not is_filtered(p):
                    g.add_node(p)

        # add tensor connections as adges to graph
        for e in conn:
            for p in conn[e].producers:
                for c in conn[e].consumers:
                    if not is_filtered(p) and not is_filtered(c):
                        g.add_edge(p, c)

        # Perform graph cleanups. Consider in future to allow to disable those cleanups.

        # remove self connections
        TorchTracer.remove_inplace_self_connections(g)
        TorchTracer.fix_inplace_bidirectional_edges(g)
        # Sometimes pytorch resuses memory between different operations.
        # This produces connections which not really exists.
        TorchTracer.prune_connections_due_to_optimization(g)
        # To improve performance of searching for all paths first prone connections via single inplace node
        TorchTracer.prune_connections_via_inplace_node(g)
        # Warning!!! Need to do 2 iterations of inplace paths removal.
        # Could be problem with algorithm and require more iterations on other models than resnet18.
        TorchTracer.prune_connections_via_inplace_path(g)
        TorchTracer.prune_connections_via_inplace_path(g)

        return g

    @staticmethod
    def remove_inplace_self_connections(graph):
        for node in graph.get_nodes():
            graph.remove_edge(node, node)

    @staticmethod
    def fix_inplace_bidirectional_edges(graph):
        nxg = graph.to_nx()
        strongly_connected = [n for n in nx.strongly_connected_components(nxg) if len(n) > 1]
        for cluster in strongly_connected:
            cluster_ops = [graph.get_node(node) for node in cluster]
            for i in range(len(cluster_ops)):
                for j in range(i + 1, len(cluster_ops)):
                    if cluster_ops[i].is_inplace or cluster_ops[j].is_inplace:
                        # op i goes before op j
                        if cluster_ops[i].idx < cluster_ops[j].idx:
                            graph.remove_edge(cluster_ops[j], cluster_ops[i])
                        else:
                            graph.remove_edge(cluster_ops[i], cluster_ops[j])

    @staticmethod
    def prune_connections_via_inplace_node(graph):
        for node in graph.get_nodes():
            for con_node in graph.gdict[node]:
                if node != con_node and con_node.is_inplace:
                    # prune connections via inplace op
                    for n in graph.gdict[con_node]:
                        if n in graph.gdict[node] and n != con_node:
                            # print("remove {}->{}".format(node.name, n.name))
                            graph.remove_edge(node, n)

    @staticmethod
    def prune_connections_via_inplace_path(graph):
        nxg = graph.to_nx()
        for node in graph.get_nodes():
            for con_node in graph.gdict[node]:
                if node != con_node and not isinstance(node, InputOp) and not isinstance(con_node, OutputOp) and not isinstance(node, ScalarOp) and not isinstance(con_node, ScalarOp):
                    max_dist = min(con_node.idx - node.idx, 10)
                    all_paths = [p for p in nx.algorithms.simple_paths.all_simple_paths(nxg, node.name, con_node.name, max_dist)
                                 if len(p) > 2]
                    for p in all_paths:
                        # if two nodes connected by inplace ops
                        inplace_path = True
                        for i in range(1, len(p) - 1):
                            inplace_path = inplace_path and graph.get_node(p[i]).is_inplace

                        # prune connection
                        if inplace_path:
                            graph.remove_edge(node, con_node)
                            break

    @staticmethod
    def prune_connections_due_to_optimization(graph):
        for (node1, node2) in graph.get_edges():
            if not isinstance(node1, InputOp) and not isinstance(node2, OutputOp) and not isinstance(node1, ScalarOp) and not isinstance(node2, ScalarOp):
                for out in node1.outputs:
                    target = [inp for inp in node2.inputs if inp.is_tensor and out.is_tensor and inp.id == out.id]
                    if len(target) > 0:
                        # prune connections created by memory reuse optimization
                        # TODO: check other inputs
                        if out.tensor_id != target[0].tensor_id:
                            graph.remove_edge(node1, node2)
                        elif out.shape != target[0].shape:
                            graph.remove_edge(node1, node2)
                        elif out.data != target[0].data:
                            graph.remove_edge(node1, node2)
