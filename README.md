# Child and Therapist Detection and Tracking
The purpose of this project is to detect all child and therapists in a video and assign a unique id against each bonding box and track them through out the video. This is a demo project for computer vision task. 

## Sample of Inferencing
![Untitled](https://github.com/user-attachments/assets/d9ae65bf-6257-4b13-a57e-1039ddf0e9f7)


## Pretrained models 
**Yolo v8** model is fine tuned on dataset given in the repository. For ID reassignment another pretrained model "**osnet_x1_0**" is used to find similarity between person in each bonding box.

## Challanges with Yolo v8
It cannot handle occlusion and also it assign new id when same person reappear in the video after some time. The proposed project overcome these challanges very well.

## Features
* Detect child and therapist and show a bonding box
* Labels each bonding box as Child or Therapist
* Assign a unique ID for each person
* It can handle occlusion, that means if a person is hiden behind other and later reappear model reassign correct id
* It also reassign correct ID if a person reappear in the video after some time

## How it works?
It works like this, save all the urls in which you want tracking in **test_video.txt** file line by line. Run the script **re_identification.py**, it will save the output video in "**output video**" folder.

## Training and Logic
First a Yolo v8 model is trained on dataset given in subfolder "**yolo_v2.1**" in google colab GPU. This model can detect child and therapist and assign a unique ID for each bonding box in each frame. For some frame this model can track person with same id but when there is occlusion it assigns new id for the same person post occlusion. And also when a person reappear appear after some time it also assign new id to the person. To handle this I have used a pretrained model "**osnet_x1_0**". This model embedd each person into a vector then python script finds similar person using cosine similarity and assign the id of most similar person to current bonding box.

## Directory Structure
* child_and_therapist_detection - Directory that contains files and directories related to training
    - Child_adult_classification colab.ipynb - Colab notebook of Yolo v8 model training
    - runs - Trained model weights saved in this folder
    - url_data.csv - Urls of videos used for training
    - yolo_v2.1 - Dataset directory
* download - saves raw video during inferencing
* output video - saves output video of each url 
* README.md - File explaing the project
* re_identification.py - Script for inferencing and also contains the logic of re_identification
* test_videos.txt - Save the url of video here in which you want detection and tracking
