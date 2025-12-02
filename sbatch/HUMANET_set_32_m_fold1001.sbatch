#! /bin/bash
#SBATCH --job-name=FASTERLTNNOBGSIALPHA
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francesco.manigrasso@polito.it 
#SBATCH --partition=cuda
#SBATCH --time=05:00:00 
#SBATCH --nodes=1 
#SBATCH --mem=20480 
#SBATCH --ntasks=8 
#SBATCH --gres=gpu:1
#SBATCH --output=/home/fmanigrasso/Tesi_moro/3DAvatarGenerator/err_log/train_%j.log 
#SBATCH --error=/home/fmanigrasso/Tesi_moro/3DAvatarGenerator/err_log/train_%j.err 
module load intel/python/3/2019.4.088 
module load nvidia/cudasdk/10.1 
cd /home/fmanigrasso/PROTOLTNendtoend 
source env2/bin/activate 
cd .. 
cd /home/fmanigrasso/Tesi_moro/3DAvatarGenerator 
python trainer_full.py --path_folder Dataset/data32/train_test_data_fold1001 --gender male --batch_size 32 --epochs 200