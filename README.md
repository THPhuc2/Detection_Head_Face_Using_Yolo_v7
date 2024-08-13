# Detection Head and Face Using YoloV7

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Training](#training)
6. [Evaluation](#evaluation)

    
![Untitled](https://github.com/user-attachments/assets/92dc5b44-02bd-4096-89ac-59df75271bfa)


## Introduction
This project focuses on detecting heads and faces using the YoloV7 architecture. The model has been customized to handle multitask learning, specifically for predicting two classes: Head and Face.


## Prerequisites
- Python 3.8+
- PyTorch
- CUDA (for GPU acceleration)
- Other dependencies are listed in the `requirements.txt`.

## Installation
1. Clone this repository:
    ```bash
    !git clone https://github.com/WongKinYiu/yolov7.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
The dataset used for this project is `scut_headface`, which includes labeled images for training and validation. 

- Download the dataset [here]([https://drive.google.com/file/d/1v5DTTaNgrBMtU60AurbUY72o9okU9jxs/view?usp=drive_link](https://drive.google.com/file/d/1abEekGDSPbuuBZAwS3IosuPGpW5e8C9P/view?usp=drive_link).
- After downloading, ensure the dataset is structured as follows:
    ```
    scut_headface/
    ├── data/
    │   ├── train/
    │   ├── trainval/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── trainval/
    │   ├── val/
    │   └── test/
    ```

## Training
To train the model with YoloV7 using the dataset:

1. Ensure the dataset is in the correct structure and create a YAML file for it:
    ```yaml
    # Khai báo 1 file yaml để YOLOv7 biết:
    # - Đường dẫn đến thư mục train, test (nếu có, nếu không thì dùng luôn đường dẫn đến train)
    # - Số lượng class qua biến nc (number of class)
    # - Tên của các class
    %cd /content/drive/MyDrive/code/Project/face_head_detection_using_Yolo_v7/yolov7
    !rm data/mydataset.yaml 
    !echo 'train: /content/drive/MyDrive/code/Project/face_head_detection_using_Yolo_v7/scut_headface/images/train' >> data/mydataset.yaml
    !echo 'val: /content/drive/MyDrive/code/Project/face_head_detection_using_Yolo_v7/scut_headface/images/val' >> data/mydataset.yaml
    !echo 'test: /content/drive/MyDrive/code/Project/face_head_detection_using_Yolo_v7/scut_headface/images/test' >> data/mydataset.yaml  # Dòng này dành cho thư mục test
    !echo 'nc: 2' >> data/mydataset.yaml
    !echo "names: ['head', 'face']" >> data/mydataset.yaml
    ```

2. Start training:
    ```bash
    python train.py --batch 8 --cfg cfg/training/yolov7.yaml --epochs 100 --data data/mydataset.yaml --weights 'pretrain/yolov7.pt'
    ```

## Evaluation
To evaluate the model on the validation or test set:

```bash
python detect.py --weights /path_to_your/trained_model.pt --source /path_to_your/scut_headface/data/test
