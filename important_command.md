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


1. convert data (no need!!)
nnUNet_convert_decathlon_task -i $nnUNet_raw_data_base/nnUNet_raw_data/Task01_STACOM_SAX
(here should be 2 digits 01,02, and program will generate a folder with 3 digits 001, 002 for us)  # see https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md

2. preprocess data
nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres -pl nnUNetPlannerResEncM -np 1


3. train 
# change batch_size = 1
nnUNetv2_train 503 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_onlyMirror01_DA5 (Add --c for continue)


4. predict (make folders first)
# change the model_best to model_final_checkpoint.model
nnUNetv2_predict_from_modelfolder -i /host/d/Data/CTA/nnUNet_raw/Dataset503_AortaProcessed/imagesTs -o /host/d/projects/aorta_seg/models/Dataset503_AortaProcessed/results/EncUNetM_3d_fullres/predicts_raw/fold_0 -m /host/d/projects/aorta_seg/models/Dataset503_AortaProcessed/nnUNetTrainer_onlyMirror01_DA5__nnUNetResEncUNetMPlans__3d_fullres -f 0




