import sys
import subprocess
import soloduration
import datetime

models=["alexnet","resnet50","vgg19","ssd"]
#models=["ssd"]

def soloduration(models):
    subprocess.getstatusoutput("")

def inithardwarepara():
    subprocess.run("./power_t_freq", shell=True)
    subprocess.run("./coninference", shell=True)

def initmodelpara():

    for i in models:
        #subprocess.getstatusoutput("./soloinference "+i)
        subprocess.run("./soloinference "+i, shell=True)
        subprocess.run("./multiinference "+i, shell=True)
        subprocess.run("./l2cache "+i, shell=True)
        #soloduration.py batch
        subprocess.run("./recordpower.sh "+i, shell=True)

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    inithardwarepara()
    t2 = datetime.datetime.now()
    initmodelpara()
    t3 = datetime.datetime.now()

    print ((t2-t1).seconds)
    print ((t3-t2).seconds)