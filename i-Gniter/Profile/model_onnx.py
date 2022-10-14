import argparse
import os

import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to generate models with.")
args = parser.parse_args()

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
print(dynamic_axes)

model = torchvision.models.alexnet().cuda()
# Fixed Shape
#torch.onnx.export(model, dummy_input, "alexnet_fixed.onnx", verbose=True, opset_version=args.opset,
 #                 input_names=input_names, output_names=output_names)
# Dynamic Shape
torch.onnx.export(model, dummy_input, "alexnet_dynamic.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)


model = torchvision.models.resnet50().cuda()
torch.onnx.export(model, dummy_input, "resnet50_dynamic.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)


model = torchvision.models.vgg19().cuda()
torch.onnx.export(model, dummy_input, "vgg19_dynamic.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)

os.system("python3 ssd_onnx.py")