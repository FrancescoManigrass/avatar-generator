
file = open("obj_test_with_std.csv","r")
list_parameters=[]
righe=0
for f in file:
    righe=righe+1
    if righe >2:
        row = f.split(";")

        source = int(row[1].split("_")[2].replace("data",""))
        dest  = int(row[1].split("_")[8].replace("data",""))
        gender = row[1].split("_")[7]
        height = round(float(row[2].replace(",", ".")), 4)
        height_std = round(float(row[3].replace(",", ".")), 4)
        weight=round(float(row[4].replace(",",".")),4)
        weight_std =round(float(row[5].replace(",",".")),4)
        chest_tot= round(float(row[6].replace(",",".")),4)
        chest_std=round(float(row[7].replace(",",".")),4)
        waist_tot = round(float(row[8].replace(",",".")),4)
        waist_std = round(float(row[9].replace(",",".")),4)
        hip_tot = round(float(row[10].replace(",", ".")), 4)
        hip_std = round(float(row[11].replace(",", ".")), 4)

        row = [source,dest,gender,height,height_std,weight,weight_std,chest_tot,chest_std,waist_tot,waist_std,hip_tot,hip_std]

        list_parameters.append(row)

gender = "male"
f= open("oldtable.txt","r")
for i in f:
    if "\\multicolumn{1}{|l|}" in i and len(i.split("&")[0].replace("}","{").split("{")) >6:





        row = i.split("&")
        if row[0]!="\\multicolumn{1}{|l|}{} ":
            source = int(row[0].replace("}","{").split("{")[10].replace("CONF-PACMO-",""))
            dest = int(row[1].replace("PACMO-",""))
        else:
            dest = int(row[1].replace("PACMO-", ""))
        for k in list_parameters:
            if k[0]==source and k[1]==dest and k[2] ==gender :



                string  = ((row[0].__str__() + "&" + row[1].__str__() + "&" + row[2].__str__() +" \\textpm   "
                           + k[4].__str__() + "&" + row[3].__str__() +" \\textpm  "+ k[6].__str__() + "&" + row[4].__str__()
                           +" \\textpm  " + k[8].__str__() + "&" + row[5].__str__() +" \\textpm  " + k[10].__str__())
                           + "&" + row[6].split("\\")[0] +  " \\textpm  " + k[12].__str__() + "  \\\\ \\cline{2-7}")


                print(string)

    else:
        print(i)

