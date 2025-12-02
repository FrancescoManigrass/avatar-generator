
possible_solutions=[10,16,32,64,128,256,300]
for i in range(len(possible_solutions)):
    for j in ["female","male"]:
        for k in [1,1001,2001,3001,4001]:
            name="HUMANET_set_"+possible_solutions[i].__str__()+"_"+j[0]+"_"+"fold"+k.__str__()+".sbatch"
            f = open(name, "w",newline="")
            f.write("#! /bin/bash\n"
                    "#SBATCH --job-name=FASTERLTNNOBGSIALPHA\n"
                    "#SBATCH --mail-type=END,FAIL\n"
                    "#SBATCH --mail-user=francesco.manigrasso@polito.it \n"
                    "#SBATCH --partition=cuda\n"
                    "#SBATCH --time=05:00:00 \n"
                    "#SBATCH --nodes=1 \n"
                    "#SBATCH --mem=20480 \n"
                    "#SBATCH --ntasks=8 \n"
                    "#SBATCH --gres=gpu:1\n"
                    "#SBATCH --output=/home/fmanigrasso/Tesi_moro/3DAvatarGenerator/err_log/train_%j.log \n"
                    "#SBATCH --error=/home/fmanigrasso/Tesi_moro/3DAvatarGenerator/err_log/train_%j.err \n"
                    "module load intel/python/3/2019.4.088 \n"
                    "module load nvidia/cudasdk/10.1 \n"
                    "cd /home/fmanigrasso/PROTOLTNendtoend \n"
                    "source env2/bin/activate \n"
                    "cd .. \n"
                    "cd /home/fmanigrasso/Tesi_moro/3DAvatarGenerator \n"
                    "python trainer_full.py --path_folder Dataset/data"+possible_solutions[i].__str__()+"/train_test_data_fold"+k.__str__()+" --gender "+j+" --batch_size 32 --epochs 200")
            f.close()