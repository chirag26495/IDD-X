# IDD-X

This repository is the official implementation of the approach proposed in [IDD-X: A Multi-View Dataset for Ego-relative Important Object Localization and Explanation in Dense and Unstructured Traffic](http://arxiv.org/abs/2404.08561). 

Please visit [Project Page](https://idd-x.github.io/) for details about the approach and the dataset.

<!-- >📋  Optional: include a graphic explaining approach
 -->

## Requirements

To install requirements (on Python 3.6.9, miniconda3 4.3.30):

(Note: *Requirements of the dependent github repos can be installed as per their stated instructions.*)

```setup
conda env create -f environment.yml
##pip install -r requirements.txt

git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout v0.24.0
git apply --whitespace=fix  ../mmaction2_tsnwcrop.patch
pip3 install -e .
cd ..

## Download tsn optical flow model pretrained on activitynet data: https://mmaction2.readthedocs.io/en/0.x/recognition_models.html#activitynet-v1-3
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth

git clone https://github.com/IDD-Detection/Yolo-v4-Model.git
mv Yolo-v4-Model  IDD-D_Yolo-v4-Model 
cd IDD-D_Yolo-v4-Model 
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
git clone https://github.com/abewley/sort.git
cd ..
```

## Download the dataset

Use this [link](https://idd.insaan.iiit.ac.in/dataset/download/) to sign up and make the download request.

Copy the downloaded dataset in this directory. Or, modify the relative data path in the respective scripts.

Refer ./dataset_README.txt for details about the dataset and its structure.

## Pre-trained Models

You can download the models trained on IDD-X here:

[OneDriveLink](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/chirag_parikh_research_iiit_ac_in/Evottwy5Q39JpDXBu7PvbngBOBPG5Xn4uC2FB3tffLJ7dA?e=hkXYJw) contains models for the tasks:
1. Important Object Track Identification
2. Ego-Vehicle's Behavior Recognition
3. Important Object Track Explanation

```
mkdir models
# Place the downloaded models inside 'models' dir
```

<!-- >Alternatively you can have an additional column in your results table with a link to the models.
 -->
 
## Data Pre-processing, Training, and Evaluation

Run the python/bash scripts in the specified order (# --> Step-#___.py/.sh)
```
# Data Preparation
python3 Step1_prepare_rawframes_dataset.py
chmod +x Step2_run.sh
bash Step2_run.sh
python3 Step3_MOTdata_extract.py
python3 Step4_associate_MOTdata_with_GT_IOtracks.py
python3 Step5_create_MOTdata_with_GT_IOtracks.py

# Train & Evaluate: Important Object Track Identification Model 
python3 Step6_train_and_evaluate_Important_Object_Selector.py

# Train & Evaluate: Ego-vehicle's Behavior Recognition Model 
python3 Step7_prepare_rawframedata_annotationsfile.py
chmod +x Step8_train_tsn_EgoVehicleBehaviorRecogizer.sh
bash Step8_train_tsn_EgoVehicleBehaviorRecogizer.sh
python3 Step9_evaluate_tsn_EgoVehicleBehaviorRecogizer.py

# Data Preparation for Important Object Explanation Generator model
python3 Step10_riskobj_mot_tracks_association_trainset.py
python3 Step11_merged_atp_mot_tracks_trainset.py
python3 Step12_prep_trackwise_headclass_explanations_data.py

# Train & Evaluate: Important Object Explanation Prediction Model 
python3 Step13_train_IOeXplanationsGeneratorModel.py
python3 Step14_evaluate_IOeXplanationsGeneratorModel.py
```

The file paths for the pre-trained models (of a. ego-vehicle behavior recognition, b. IO eXplanation generator) specified in the scripts (Step13_train_IOeXplanationsGeneratorModel.py and Step14_evaluate_IOeXplanationsGeneratorModel.py) can be replaced with the best models obtained after running their respective training scripts.

<!-- ## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
 -->

## Citation.
If you find the code and dataset useful, please cite this paper (and refer the data as IDD-X):
```
@misc{parikh2024iddx,
      title={IDD-X: A Multi-View Dataset for Ego-relative Important Object Localization and Explanation in Dense and Unstructured Traffic}, 
      author={Chirag Parikh and Rohit Saluja and C. V. Jawahar and Ravi Kiran Sarvadevabhatla},
      year={2024},
      eprint={2404.08561},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References

We thank the following github repositories for the relevant open source code and models:

**[MMAction2](https://github.com/open-mmlab/mmaction2)**

**[IDD-Detection Model](https://github.com/IDD-Detection/Yolo-v4-Model)**

**[SORT Tracker](https://github.com/abewley/sort)**

**[Blurring Faces & License Plates in Dashcam videos](https://github.com/varungupta31/dashcam_anonymizer)**


