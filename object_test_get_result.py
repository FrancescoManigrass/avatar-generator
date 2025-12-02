import csv

f = open('obj_test2.csv', 'w',newline="")

# create the csv writer
writer = csv.writer(f,delimiter=";")


row=["folder1","folder2","counter","height_tot","weight_tot","chest_tot","waist_tot","hip_tot"]

writer.writerow(row)

f = open("output_object_test.txt", "r",encoding='UTF8')
trovato=0
correct_elements=[]
for x in f:
    if "folder1" in x:
        trovato=1
        correct_elements = []
    if trovato==1:
        if "folder1" in x:
            correct_elements.append(x.split("folder1")[1].rstrip())
        if "folder2" in x:
            correct_elements.append(x.split("folder2")[1].rstrip().split("\\")[-1])
        if "cpunter value is:" in x:
            correct_elements.append(int(x.split("cpunter value is:")[1]))

        if "mean height error" in x:
            correct_elements.append(round(float(x.split("mean height error:")[1]),4))
        if "mean weight error" in x:
            correct_elements.append(round(float(x.split("mean weight error:")[1]),4))
        if "mean chest error:" in x:
            correct_elements.append(round(float(x.split("mean chest error:")[1]),4))
        if "mean waist error:" in x:
            correct_elements.append(round(float(x.split("mean waist error:")[1]),4))
        if "mean hip error:" in x:
            correct_elements.append(round(float(x.split("mean hip error:")[1]),4))

            trovato=0
            writer.writerow(correct_elements)






