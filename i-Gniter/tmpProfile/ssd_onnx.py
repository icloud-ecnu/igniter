
import argparse
import torch
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to generate models with.")
args = parser.parse_args()

dummy_input = torch.randn(10, 3, 300, 300, device='cuda')
#model = torchvision.models.alexnet().cuda()
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').cuda()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Fixed Shape
#torch.onnx.export(model, dummy_input, "alexnet_fixed.onnx", verbose=True, opset_version=args.opset,
 #                 input_names=input_names, output_names=output_names)

# Dynamic Shape
dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
print(dynamic_axes)
torch.onnx.export(ssd_model, dummy_input, "ssd_dynamic.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)
