import os

os.system("rm -rf force_algorithm_config.txt")
SLOs = [[10,20,20,25],[15,30,30,40],[20,40,40,55]]
Rates = [[1200,400,300,150],[400,600,400,50],[800,200,200,300]]

for i in range(len(SLOs)):
    slos = ""
    rates = ""
    for x in SLOs[i]:
        if len(slos) > 0: slos+=":"
        slos += str(x)
    for x in Rates[i]:
        if len(rates) > 0: rates += ":"
        rates += str(x)
    script = "python3 force_algorithm.py -s {} -r {}".format(slos,rates)
    print(script)
    os.system(script)
