import sys
import json

def listaverage(l,s,e):
    #print(l,s,e)
    asum=0
    for i in range(s,e):
        asum+=l[i]
    return asum/(e-s)

def saverecords(kind,metrics,value):
    path="config"
    #path="config_test"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

def dmonps(openfilepath,kind):
    metrics=["sm","pwr","pclk"]
    dirmetrics={}
    #dirmetrics[metrics]["data"][value]
    for i in metrics:
        dirmetrics[i]={}
        dirmetrics[i]["d"]=0
        dirmetrics[i]["data"]=[]
    with open(openfilepath, encoding='utf-8') as f:
        log=f.readlines()
        line1=log[0].split()
        for i in metrics:
            dirmetrics[i]["d"]=line1.index(i)-1
        for i in range(2,len(log)):
            lines = log[i].split()
            for j in metrics:
                if(lines[0]!="#"):
                    try:
                        dirmetrics[j]["data"].append(int(lines[dirmetrics[j]["d"]]))
                    except IndexError:
                        pass
    segs=[]
    flag=0
    index = 0
    idlepower=(dirmetrics["pwr"]["data"][0]+dirmetrics["pwr"]["data"][1])/2
    for i in dirmetrics["sm"]["data"]:
        if(flag==0 and i>=90):
            segs.append([index+1,0])
            flag=1
            #print(i)
        if(flag==1 and i<90):
            segs[-1][1]=index-1
            flag=0
        index += 1
    if(flag==1):
        segs[-1][1]=index-1
        flag=0
    #print(segs)
    for i in metrics:
        values=[[]]
        value=[]
        if(i!="sm"):
            for seg in segs:
                #print(listaverage(dirmetrics[i]["data"],seg[0],seg[1]))
                a=listaverage(dirmetrics[i]["data"],seg[0],seg[1])
                if(a<10):
                    continue
                if(len(values[-1])>=3):
                    values.append([])
                values[-1].append(a)
                value.append(a)
                print(a)
        if(kind=="sys"):
            if(i=="pwr"):
                saverecords("hardware","idlepower",idlepower)
                saverecords("hardware","power",value)
            elif(i=="pclk"):
                saverecords("hardware","frequency",value)
        elif(openfilepath[-1]=="2"):
            if(i=="pwr"):
                saverecords(kind,"power_2",value[0])
            elif(i=="pclk"):
                saverecords(kind,"frequency_2",value[0])
        else:
            if(i=="pwr"):
                saverecords(kind,"power",values)
            elif(i=="pclk"):
                saverecords(kind,"frequency",values)

if __name__ == '__main__':
    dmonps(sys.argv[1],sys.argv[2])
