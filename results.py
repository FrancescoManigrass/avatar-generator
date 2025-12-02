
file = open("3DAvatarGenerator_paper.csv" ,"r")
rows =0
mydict ={}
title =""
for f in file:
    if rows != 0:
        elements = f.strip().split(",")
        if elements[5] !="" and int(elements[13] )==200:


            tmp_dict ={}
            tmp_dict["val_3d_shape_mean" ] =float(elements[5])
            tmp_dict["val_3d_shape_std" ] =float(elements[6])
            tmp_dict["val_3d_parameter_mean"] = float(elements[7])
            tmp_dict["val_3d_parameter_std"] = float(elements[8])

            tmp_dict["test_3d_shape_mean"] = float(elements[9])
            tmp_dict["test_3d_shape_std"] = float(elements[10])
            tmp_dict["test_3d_parameter_mean"] = float(elements[11])
            tmp_dict["test_3d_parameter_std"] = float(elements[12])

            if elements[3]+"_"+elements[4] not in mydict:

                mydict[elements[3]+"_"+elements[4]] =tmp_dict

    else:
        title = f.strip().split(",")

    rows += 1
final_results ={}
final_results_test ={}
for i in [10 ,16 ,32 ,64 ,128 ,256 ,300]:

    for g in ["male","female"]:
        tmp_dict = {'val_3d_shape_mean': 0, 'val_3d_shape_std': 0, 'val_3d_parameter_mean': 0,
                    'val_3d_parameter_std': 0}




        for f in mydict:
            if "data" +i.__str__() in f and f.split("_")[-1] == g:

                tmp_dict["val_3d_shape_mean" ] += mydict[f]["val_3d_shape_mean"]
                tmp_dict["val_3d_shape_std"] += mydict[f]["val_3d_shape_std"]
                tmp_dict["val_3d_parameter_mean"] += mydict[f]["val_3d_parameter_mean"]
                tmp_dict["val_3d_parameter_std"] += mydict[f]["val_3d_parameter_std"]

                if f.split("/")[1]+"_"+g not in final_results_test:
                    final_results_test[f.split("/")[1]+"_"+g] = mydict[f]
                    final_results_test[f.split("/")[1]+"_"+g]["id"] = f
                else:
                    if final_results_test[f.split("/")[1]+"_"+g]['test_3d_shape_mean'] < mydict[f]['test_3d_shape_mean']:
                        final_results_test[f.split("/")[1]+"_"+g] = mydict[f]
                        final_results_test[f.split("/")[1]+"_"+g]["id"] = f








        tmp_dict["val_3d_shape_mean"] /= 5
        tmp_dict["val_3d_shape_std"] /= 5
        tmp_dict["val_3d_parameter_mean"] /= 5
        tmp_dict["val_3d_parameter_std"] /= 5
        final_results["data " +i.__str__()+"_"+g ] =tmp_dict



cont=0
for i in final_results:
    if cont%2==0:
        print("PACMO-"+i.split("_")[0].split(" ")[1],"&",round(final_results[i]['val_3d_shape_mean'],5), "\\textpm",round(final_results[i]['val_3d_shape_std'],5),
              "&",round(final_results[i]['val_3d_parameter_mean'],5), "\\textpm",round(final_results[i]['val_3d_parameter_std'],5) , "\\", end="")
    else:
        print( "&", round(final_results[i]['val_3d_shape_mean'], 5), "\\textpm",
              round(final_results[i]['val_3d_shape_std'], 5),
              "&", round(final_results[i]['val_3d_parameter_mean'], 5), "\\textpm",
              round(final_results[i]['val_3d_parameter_std'], 5), "\\")
    cont+=1

print("BEST TEST MODELS")
for f in final_results_test:
    print(final_results_test[f]["id"])
