import os

files = os.listdir(".")
for f in files:
    if "HUMANET_set_10" in f:
        os.system("bash -c 'sbatch %s'" % f)