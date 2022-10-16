import argparse
import torch
import torchvision
from transformers import BertModel
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction

parser = argparse.ArgumentParser()
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to generate models with.")
args = parser.parse_args()

Batch_size=64
seg_length=128
dummy_input0 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
dummy_input1 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
dummy_input2 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
#model = torchvision.models.vgg19(pretrained=True).cuda()
modelConfig = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',config=modelConfig).cuda()

input_names = [ "input_ids", "attention_mask", "token_type" ] #+ [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1","output2" ]

#input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
#output_names = [ "output1" ]

# Fixed Shape
torch.onnx.export(model, (dummy_input0, dummy_input1, dummy_input2), "bert_fixed_b1.onnx", verbose=True, opset_version=args.opset)
                 #input_names=input_names, output_names=output_names)

# Dynamic Shape
#dynamic_axes = {"input_ids":{0:"batch_size"}, "attention_mask":{0:"batch_size"}, "token_type":{0:"batch_size"},"output1":{0:"batch_size"}, "output2":{0:"batch_size"}}
#dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
#print(dynamic_axes)
#torch.onnx.export(model,(dummy_input0, dummy_input1, dummy_input2), "bert_dynamic_2.onnx", verbose=True, opset_version=args.opset,input_names=input_names, output_names=output_names,dynamic_axes=dynamic_axes)
