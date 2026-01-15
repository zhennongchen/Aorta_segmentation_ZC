# Segment ascending aorta using nnUNetRecEncUNet
**Author: Zhennong Chen, Xi'an Jiaotong-Liverpool University, 2026**

step 0: install nnUNet
do it in the terminal as the root user of docker 
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

step 1: in the terminal as the root user of docker, type the following pathes
export nnUNet_raw="/host/d/Data/CTA/nnUNet_raw"
export nnUNet_preprocessed="/host/d/Data/CTA/nnUNet_preprocessed"
export nnUNet_results="/host/d/projects/aorta_seg/models"
export nnUNet_compile=0

step 2: preprocessing the data using image_preprocessing.ipynb

step 3: prepare the nnUNet dataset using prepare_nnunet_data.ipynb
- For our dataset (TAA), i call it Dataset504_AortaTAA
- For public dataset, i call it Dataset503_AortaProcessed

step 4. preprocess data for nnUNet experiments
- in the terminal, type:
   nnUNetv2_plan_and_preprocess -d 504 -c 3d_fullres -pl nnUNetPlannerResEncM -np 1
- in the generated text file (nnUNetResEncUNetMPlans.txt), change the batch_size to 1 if GPU memory is limited.


step 5. train 
- in the terminal, type:
   nnUNetv2_train 504 3d_fullres fold -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_onlyMirror01_DA5 
- fold can be 0,1,2,3,4, empirically, each fold generates similar results
- if fine-tuned from the model trained on public dataset, add the argument: -pretrained_weights /host/d/projects/aorta_seg/models/Dataset503_AortaProcessed/checkpoint_final_fold0.pth(replace with your own path)  
- if need to continue training (for example you accidentally interrupt the trianing), Add --c 


step 6. predict 
- in the model folder (nUNet_results), manually change the checkpoint_best.pth to checkpoint_final.pth if there is no checkpoint_final.pth
- make folders where you are going to save the prediction results, e.g.,
  /host/d/projects/aorta_seg/models/Dataset504_AortaTAA/results/EncUNetM_3d_fullres/predicts_raw/fold_0
- in the terminal, type:
nnUNetv2_predict_from_modelfolder -i /host/d/Data/CTA/nnUNet_raw/Dataset504_AortaTAA/imagesTs -o /host/d/projects/aorta_seg/models/Dataset504_AortaTAA/results/EncUNetM_3d_fullres/predicts_raw/fold_0 -m /host/d/projects/aorta_seg/models/Dataset504_AortaTAA/nnUNetTrainer_onlyMirror01_DA5__nnUNetResEncUNetMPlans__3d_fullres -f 0

step 7. post-processing and quantitative analysis using post_processing.ipynb




