import os

init=556416
end=556426
index = [f for f in range(init,end)]
for f in index:

    os.system("bash -c 'scancel %s'" % f.__str__())