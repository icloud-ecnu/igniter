import gslice
import subprocess


def testans(inference,resource,batch):
    l=len(inference)
    index=100001
    shell="./testinference.sh "+str(index)+" "
    for i in range(l):
        # resource 25
        shell+=(inference[i]+" "+str(resource[i])+" "+str(batch[i])+" ")
    subprocess.run(shell, shell=True)
    observe=[]
    j=1
    for i in range(l):
        o=gslice.durationps("data/durtime_"+str(index)+"_"+inference[i]+"_"+str(j))[0]
        j+=1
        o[0]*=batch[i]
        observe.append(o)
    print(observe)

if __name__ == '__main__':
    models=["alexnet","resnet50","vgg19","ssd"]
    #testans([models[0],models[1]],[17.5,35],[5,10])
    #testans([models[0],models[1],models[2]],[15,32.5,37.5],[10,8,8])
    #testans([models[0],models[3]],[25,75],[19,3])
    testans([models[3]],[80],[5])
