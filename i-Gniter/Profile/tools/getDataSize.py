import json


def saverecords(kind,metrics,value):
    path="config"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")


models = ["alexnet","resnet50","vgg19","ssd"]
inputData = [602112,602112,602112,1080000]
outputData = [4000,4000,4000,2968880]
for i in range(len(models)):
    saverecords(models[i],"inputdata",inputData[i])
    saverecords(models[i],"outputdata",outputData[i])