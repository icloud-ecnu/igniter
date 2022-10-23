import os

cmd = "./computeBandwidth.sh > PCIeInfo.txt"
os.system(cmd)
os.system("python3 collectPCIe.py")

