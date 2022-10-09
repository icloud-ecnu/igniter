import json

def loadprofile(path):
    profile={}
    with open(path,"r") as file:
        for line in file:
            data=json.loads(line)
            for key in data:
                #profile.update(data)
                if(key not in profile):
                    profile[key]=data[key]
                    #profile[key]=copy.deepcopy(data[key])
                else:
                    #print(profile[key],data[key])
                    profile[key].update(data[key])
                #print(profile)
    #print(profile)
    return profile

def saverecords(type,metrics,value):
    path="testcon_trans.log"
    records={type:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

if __name__ == '__main__':
    profile=loadprofile("./config_test")
    batches=[3,9,27]
    models=["alexnet","resnet50","vgg19","ssd"]
    threads=[20,40,50,60,80]
    gpu_duration=[]
    inference_duration=[]
    metrics=["gpulatency","inferencelatency","power","frequency"]
    power=[]
    frequency=[]
    #print(mpsid)
    k=0
    for m in models:
        for me in metrics:
            values=[]
            for i in profile[m][me]:
                for j in i:
                    values.append(j)
            ans={}
            k=0
            for i in batches:
                ans[i]={}
                for j in threads:
                    ans[i][j]=values[k]
                    k+=1
            saverecords(m,me,ans)
