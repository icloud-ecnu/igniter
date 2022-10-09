import subprocess
import os
import sys
import trtexecps
import json

def saverecords(kind,metrics,value):
    path="config"
    #path="config_test"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

def soloduration(model):
    batches=[1,16,32]
    threads=[10,50,100]
    #batches=[3,9,27]
    #threads=[20,40,50,60,80]
    #mpsid=os.getenv('MPSID')
    gpu_duration=[]
    inference_duration=[]
    #print(mpsid)
    k=0
    for i in batches:
        gpu_duration.append([])
        inference_duration.append([])
        for j in threads:
            subprocess.run("./soloduration "+model+" "+str(j)+" "+str(i), shell=True)
            outputpath="data/durtime_"+model+"_b"+str(i)+"_t"+str(j)
            dur=trtexecps.trtexecps(outputpath)
            gpu_duration[k].append(dur[0])
            inference_duration[k].append(dur[1])
        k+=1
    if(model=="vgg19"):
        vggpath="data/durtime_vgg19_b"+str(batches[-1])+"_t"+str(threads[-1])
        htddur=trtexecps.trtexec(vggpath,"h2dMs")
        saverecords(model,"htddur",[batches[-1],htddur])
    saverecords(model,"gpulatency",gpu_duration)
    saverecords(model,"inferencelatency",inference_duration)

if __name__ == '__main__':
    soloduration(sys.argv[1])

