import torch
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from torchvision.models import mobilenet_v2
from executorch.exir import EdgeProgramManager, to_edge, ExecutorchProgramManager
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

NUM_CLASSES = 10

mobilenet = mobilenet_v2()
mobilenet.classifier[1] = nn.Linear(in_features=1280, out_features=NUM_CLASSES, bias=True)
mobilenet.load_state_dict(torch.load("./110550059_model.pt", map_location='cpu'))
mobilenet.eval()
example_args = (torch.randn(1, 3, 224, 224), )
pre_autograd_aten_dialect = capture_pre_autograd_graph(mobilenet, example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
print(aten_dialect)

edge_program: EdgeProgramManager = to_edge(aten_dialect)

# delegate to xnn
edge_program = edge_program.to_backend(XnnpackPartitioner)

executorch_program: ExecutorchProgramManager = edge_program.to_executorch()

# Serialize and save it to a file
save_path = "xnn_mobilenet.pte"
with open(save_path, "wb") as f:
    f.write(executorch_program.buffer)
