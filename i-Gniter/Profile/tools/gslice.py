# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess
import math
from numpy.lib.function_base import copy
import numpy as np
import logging


MAX_BATCH=32
MAX_RESOURCE=100
def durationps(openfilepath):
    # numbers, inference latency
    with open(openfilepath, encoding='utf-8') as f:
        log=f.readlines()
        gpulatency=[]
        inferencelatency=[]
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>10000 and j["endComputeMs"]<20000):
                #latencyMs computeMs
                gpulatency.append(j["computeMs"])
                inferencelatency.append(j["latencyMs"])
        if(len(gpulatency)>10):
            #print(sum/n,sum2/n)
            return([len(gpulatency)/10,np.mean(inferencelatency)],[np.std(gpulatency),np.std(inferencelatency)])
        else:
            print("Duration time is too short!")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def compute_new_gpu_util(current_gpu_util, slo, arrival_rate, avg_latency, avg_throughput):
    """
    :param current_gpu_util: 当前gpu使用量
    :param slo: 服务水平
    :param arrival_rate:  要满足的吞吐量要求
    :param avg_latency: 平均延时
    :param avg_throughput: 平均吞吐量
    :return: 调整后的gpu使用量
    """
    residual_latency = slo - avg_latency  # 剩余延时能力，正常情况avg应该小于slo
    residual_throughput = avg_throughput - arrival_rate  # 剩余吞吐量，正常情况下avg应该大于arrival_rate
    diff_latency = residual_latency * 100 / slo  # 剩余延时比例
    diff_throughput = residual_throughput * 100 / arrival_rate  # 剩余吞吐量比例
    if diff_latency > 0 and diff_latency < 10 and diff_throughput > 0 and diff_throughput < 10:  # 剩余的量不多，就直接返回当前资源量
        return current_gpu_util
    change_factor = max(abs(residual_latency) / avg_latency, abs(residual_throughput) / avg_throughput)  # 调整比例
    change_gpu_util = current_gpu_util * change_factor  # 调整的资源量
    new_gpu_util = 0
    if residual_latency < 0 or residual_throughput < 0:  # 剩余量出现负值，代表资源不够，因此需要增加资源
        new_gpu_util = current_gpu_util + change_gpu_util
    if residual_latency > 0 and residual_throughput > 0:  # 剩余量均为正值，代表资源满足要求，可以适当减少资源
        new_gpu_util = current_gpu_util - change_gpu_util
    if(new_gpu_util>MAX_RESOURCE):
        return MAX_RESOURCE
    elif(new_gpu_util<1):
        return 1
    return new_gpu_util


def is_time_to_adjust(slo, avg_latency, cur_batch):
    return True


def slab(slo, avg_latency, cur_batch):
    residual_latency = slo - avg_latency  # 剩余延时能力，正常情况avg应该小于slo
    diff_latency = residual_latency * 100 / slo  # 剩余延时比例
    if diff_latency > 0 and diff_latency < 10:
        return cur_batch
    change_batch = abs(residual_latency) / avg_latency * cur_batch
    new_batch = 1
    if residual_latency > 0:
        new_batch = cur_batch + change_batch
    elif residual_latency < 0:
        new_batch = cur_batch - change_batch
    #print("batch",cur_batch,new_batch,slo, avg_latency)
    if(new_batch>MAX_BATCH):
        new_batch=MAX_BATCH
    elif(new_batch<1):
        new_batch=1
    return math.ceil(new_batch)


def update_batch(new_batch):
    print("set the new batch as {0}".format(new_batch))


def update_gpu_util(new_gpu_util):
    print("set the new gpu utilization as {0}".format(new_gpu_util))


class Infere:
    batch=[]
    resource=[]
    latency=[]
    throughput=[]
    def printi(self):
        print("batch ",self.batch)
        print("resource ",self.resource)
        print("latency ",self.latency)
        print("throughput ",self.throughput)
        print()
    def printth(self):
        print("batch ",self.batch)
        print("latency ",self.latency)
        th=[]
        for i in range(len(self.throughput)):
            th.append(self.batch[i]*self.throughput[i])
        print("throughput ",th)
        print()
    def __init__(self) -> None:
        self.batch=[]
        self.resource=[]
        self.latency=[]
        self.throughput=[]

def gslice(inference,arrivalrate,slo,index):
    l=len(arrivalrate)
    batch=[1]*l
    resource=[30]*l
    #index 100 200
    inferes=[]
    for i in range(l):
        i1=Infere()
        i1.batch.append(1)
        i1.resource.append(30)
        inferes.append(i1)
    print(inferes)

    for p in range(5):
        for t in range(5):
            shell="./testinference.sh "+str(index)+" "
            for i in range(l):
                # resource 25
                shell+=(inference[i]+" "+str(resource[i])+" "+str(batch[i])+" ")
            logging.info("%s",shell)
            subprocess.run(shell, shell=True)
            flag=True
            for i in range(l):
                observe=durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
                inferes[i].latency.append(observe[1])
                inferes[i].throughput.append(observe[0])
                curbatch=batch[i]
                batch[i]=slab(slo[i],observe[1],batch[i])
                #logging.info("avg_latency %s",str(observe[1]))
                #logging.info("SLAB %s %s",str(batch[i]),str(resource[i]))
                inferes[i].batch.append(batch[i])
                inferes[i].resource.append(resource[i])
                if(batch[i]!=curbatch):
                    flag=False
            index+=1
            if(flag):
                break
        shell="./testinference.sh "+str(index)+" "
        for i in range(l):
            # resource 25
            shell+=(inference[i]+" "+str(resource[i])+" "+str(batch[i])+" ")
        logging.info("%s",shell)
        subprocess.run(shell, shell=True)
        flag=True
        for i in range(l):
            observe=durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
            inferes[i].latency.append(observe[1])
            inferes[i].throughput.append(observe[0])
            curresource=resource[i]
            resource[i]=compute_new_gpu_util(resource[i],slo[i],arrivalrate[i],observe[1],batch[i]*observe[0])
            #logging.info("observe %s",str(observe))
            #logging.info("resource %s %s",str(batch[i]),str(resource[i]))
            inferes[i].batch.append(batch[i])
            inferes[i].resource.append(resource[i])
            if(resource[i]!=curresource):
                flag=False
        index+=1
        if(flag):
            break
    for t in range(5):
        shell="./testinference.sh "+str(index)+" "
        for i in range(l):
            # resource 25
            shell+=(inference[i]+" "+str(resource[i])+" "+str(batch[i])+" ")
        logging.info("%s",shell)
        subprocess.run(shell, shell=True)
        flag=True
        for i in range(l):
            observe=durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
            inferes[i].latency.append(observe[1])
            inferes[i].throughput.append(observe[0])
            curbatch=batch[i]
            batch[i]=slab(slo[i],observe[1],batch[i])
            #logging.info("avg_latency %s",str(observe[1]))
            #logging.info("SLAB %s %s",str(batch[i]),str(resource[i]))
            inferes[i].batch.append(batch[i])
            inferes[i].resource.append(resource[i])
            if(batch[i]!=curbatch):
                flag=False
        index+=1
        if(flag):
            break
    shell="./testinference.sh "+str(index)+" "
    for i in range(l):
        # resource 25
        shell+=(inference[i]+" "+str(resource[i])+" "+str(batch[i])+" ")
    subprocess.run(shell, shell=True)
    for i in range(l):
        observe=durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
        inferes[i].latency.append(observe[1])
        inferes[i].throughput.append(observe[0])
    for i in inferes:
        for j in range(len(i.batch)):
            i.throughput[j]*=i.batch[j]
        i.printi()   
    return [batch,resource]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S', filename='gslice3.log', filemode='a')
    models=["alexnet","resnet50","vgg19","ssd"]
    #g=gslice([models[0],models[1]],[1000,500],[5,20])
    g=gslice([models[2],models[3]],[300,100],[10,20],150)
    print(g)
