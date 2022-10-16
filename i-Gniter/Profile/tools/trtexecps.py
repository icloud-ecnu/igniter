import json
import sys
def trtexecps(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        
        log=f.readlines()
        sum = 0
        sum2 =0
        n = 0
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>2000 and j["endComputeMs"]<8000):
                #latencyMs computeMs
                sum += j["computeMs"]
                sum2 += j["latencyMs"]
                n += 1
        if(n!=0):
            print(sum/n,sum2/n)
            return([sum/n,sum2/n])
        else:
            print("Duration time is too short!")


def trtexec(openfilepath, type):
    # type: latency type
    with open(openfilepath, encoding='utf-8') as f:
        log=f.readlines()
        sum = 0
        n = 0
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>2000 and j["endComputeMs"]<8000):
                #latencyMs computeMs
                sum += j[type]
                n += 1
        if(n!=0):
            print(sum/n)
            return(sum/n)

def trtexecps_th(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        
        log=f.readlines()
        sum = 0
        sum2 =0
        n = 0
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>1000 and j["endComputeMs"]<4000):
                #latencyMs computeMs
                #sum += j["computeMs"]
                sum2 += j["latencyMs"]
                n += 1
        if(n!=0):
            #print(sum/n,sum2/n)
            return([n/3,sum2/n])
        else:
            print("Duration time is too short!")

if __name__ == '__main__':
    trtexecps(sys.argv[1])
