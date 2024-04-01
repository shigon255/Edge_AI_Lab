import torch
import torch_pruning as tp
from torchvision.models import mobilenet_v2

model = torch.load('./model/mobilenet.pth', map_location="cpu")
example_inputs = torch.zeros(1, 3, 224, 224).to(args.device)
ops, size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)

MFLOPs = ops/1e6
print(MFLOPs)