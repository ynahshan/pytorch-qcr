import unittest
import torch
from torchvision import models
from pyqcr.torchgraph.tracer import TorchTracer


resnet18_graph = {'Conv2d0': ['BatchNorm2d0'], 'Tensor0': ['Conv2d0'], 'BatchNorm2d0': ['ReLU0'], 'ReLU0': ['MaxPool2d0'], 'MaxPool2d0': ['Conv2d1', '__iadd__0'], 'Conv2d1': ['BatchNorm2d1'], '__iadd__0': ['ReLU2'], 'BatchNorm2d1': ['ReLU1'], 'ReLU1': ['Conv2d2'], 'Conv2d2': ['BatchNorm2d2'], 'BatchNorm2d2': ['__iadd__0'], 'ReLU2': ['Conv2d3', '__iadd__1'], 'Conv2d3': ['BatchNorm2d3'], '__iadd__1': ['ReLU4'], 'BatchNorm2d3': ['ReLU3'], 'ReLU3': ['Conv2d4'], 'Conv2d4': ['BatchNorm2d4'], 'BatchNorm2d4': ['__iadd__1'], 'ReLU4': ['Conv2d5', 'Conv2d7'], 'Conv2d5': ['BatchNorm2d5'], 'Conv2d7': ['BatchNorm2d7'], 'BatchNorm2d5': ['ReLU5'], 'ReLU5': ['Conv2d6'], 'Conv2d6': ['BatchNorm2d6'], 'BatchNorm2d6': ['__iadd__2'], '__iadd__2': ['ReLU6'], 'ReLU6': ['Conv2d8', '__iadd__3'], 'Conv2d8': ['BatchNorm2d8'], '__iadd__3': ['ReLU8'], 'BatchNorm2d7': ['__iadd__2'], 'BatchNorm2d8': ['ReLU7'], 'ReLU7': ['Conv2d9'], 'Conv2d9': ['BatchNorm2d9'], 'BatchNorm2d9': ['__iadd__3'], 'ReLU8': ['Conv2d10', 'Conv2d12'], 'Conv2d10': ['BatchNorm2d10'], 'Conv2d12': ['BatchNorm2d12'], 'BatchNorm2d10': ['ReLU9'], 'ReLU9': ['Conv2d11'], 'Conv2d11': ['BatchNorm2d11'], 'BatchNorm2d11': ['__iadd__4'], '__iadd__4': ['ReLU10'], 'ReLU10': ['Conv2d13', '__iadd__5'], 'Conv2d13': ['BatchNorm2d13'], '__iadd__5': ['ReLU12'], 'BatchNorm2d12': ['__iadd__4'], 'BatchNorm2d15': ['ReLU13'], 'Conv2d15': ['BatchNorm2d15'], 'BatchNorm2d13': ['ReLU11'], 'ReLU11': ['Conv2d14'], 'Conv2d14': ['BatchNorm2d14'], 'BatchNorm2d14': ['__iadd__5'], 'ReLU12': ['Conv2d15', 'Conv2d17'], 'Conv2d17': ['BatchNorm2d17'], 'ReLU13': ['Conv2d16'], 'Conv2d16': ['BatchNorm2d16'], 'BatchNorm2d16': ['__iadd__6'], '__iadd__6': ['ReLU14'], 'ReLU14': ['Conv2d18', '__iadd__7'], 'Conv2d18': ['BatchNorm2d18'], '__iadd__7': ['ReLU16'], 'BatchNorm2d17': ['__iadd__6'], 'ReLU15': ['Conv2d19'], 'Conv2d19': ['BatchNorm2d19'], 'BatchNorm2d18': ['ReLU15'], 'BatchNorm2d19': ['__iadd__7'], 'ReLU16': ['AdaptiveAvgPool2d0'], 'AdaptiveAvgPool2d0': ['flatten0'], 'flatten0': ['Linear0'], 'Linear0': ['Tensor1'], 'const0': ['flatten0'], 'Tensor1': []}


class TestTracer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_resnet18_graph(self):
        model = models.resnet18()
        inp = torch.rand((1, 3, 224, 224))
        with TorchTracer() as tt:
            tt.trace_model(model, inp)

        g = tt.to_graph()
        ng = g.to_namegraph()
        for k in ng.gdict:
            self.assertTrue(k in resnet18_graph)
            self.assertEqual(ng.gdict[k], resnet18_graph[k])
