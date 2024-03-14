import torch
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from torchvision.models import mobilenet_v2
from executorch.exir import EdgeProgramManager, to_edge, ExecutorchProgramManager, ExecutorchBackendConfig


example_args = (torch.randn(1, 3, 224, 224), )
pre_autograd_aten_dialect = capture_pre_autograd_graph(mobilenet_v2().eval(), example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
print(aten_dialect)

edge_program: EdgeProgramManager = to_edge(aten_dialect)
executorch_program: ExecutorchProgramManager = edge_program.to_executorch()

# Serialize and save it to a file
save_path = "mobilenet.pte"
with open(save_path, "wb") as f:
    f.write(executorch_program.buffer)
