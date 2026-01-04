install:
git clone https://github.com/MIC-DKFZ/nnUNet.git
# do it in the root user of docker
cd nnUNet
git checkout nnunetv1
pip install -e .

export nnUNet_raw="/host/d/Data/CTA/nnUNet_raw"
export nnUNet_preprocessed="/host/d/Data/CTA/nnUNet_preprocessed"
export nnUNet_results="/host/d/projects/aorta_seg/models"
export nnUNet_compile=0

1. convert data
nnUNet_convert_decathlon_task -i $nnUNet_raw_data_base/nnUNet_raw_data/Task01_STACOM_SAX
(here should be 2 digits 01,02, and program will generate a folder with 3 digits 001, 002 for us)  # see https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md

2. preprocess data
nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres -pl nnUNetPlannerResEncL -np 16 

#nnUNet_plan_and_preprocess -t 1

3. train 
nnUNet_train 3d_fullres nnUNetTrainerV2 1 0  # 1 is the task id, 0 is the fold id  (--continue_training if continue)

4. predict (make folders first)
# change the model_best to model_final_checkpoint.model
nnUNet_predict -i /mnt/camca_NAS/SAM_for_CMR/data/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_STACOM_SAX/imagesTs -o /mnt/camca_NAS/SAM_for_CMR/models/nnUNet/predicts_raw/Task001_STACOM_SAX -t 1 -m 3d_fullres

nnUNet_predict -i /mnt/camca_NAS/SAM_for_CMR/data/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task013_MM_zeroshot/imagesTs -o /mnt/camca_NAS/SAM_for_CMR/models/nnUNet/predicts_raw/Task013_MM_zeroshot -t 13 -m 3d_fullres



