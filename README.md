# YOLO-Based Bus Road-Scene Recognition

## Overview

This project detects and classifies road-scene objects in bus-mounted video. The
complete system combines a YOLO detector, several ResNet classifiers, and a
Streamlit interface. Example target categories include traffic-light structures,
light appearance, directional arrows, and pedestrian crossings.

This project has a less restrictive code boundary than the research projects in
the other directories. However, all company data and business-specific details
remain private.

## Dataset

### Private Bus-Company Data

The training data consists of road video collected by a partner bus company. It
is a private, non-public dataset. There is no public download link.


## System Pipeline

1. YOLOv8 detects road-scene objects.
2. Detected regions of interest are cropped by object category.
3. ResNet18 classifiers refine selected detections.
4. A Streamlit application displays predictions for user-provided images.

The public scripts use command-line arguments or interface inputs and do not
contain local data paths.

## Environment

### Verified Software Configuration

- Linux x86_64;
- Python 3.9.23;
- PyTorch 2.5.1 with CUDA 12.1;
- Torchvision 0.20.1 with CUDA 12.1;
- Ultralytics 8.3.229;
- Streamlit 1.50.0;
- OpenCV 4.9.0;
- NumPy 1.23.5.

### Conda Installation

```bash
conda env create -f environment.yml
conda activate bus-road-yolo
```

### Minimal Virtual-Environment Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Hardware

The project was developed and trained on a Linux workstation with:

- 4 × NVIDIA GeForce RTX 3090 GPUs, 24 GB per GPU;
- 2 × Intel Xeon E5-2680 v4 CPUs, 28 cores and 56 threads in total;
- approximately 188 GB of system memory;
- NVIDIA driver 530.30.02.

YOLO and the ResNet classifiers can be trained on a single GPU. Multiple GPUs
were used to run independent classification experiments in parallel. The
Streamlit interface can run on a CPU, although GPU inference is faster.

## Usage

### Train the YOLO Detector

```bash
python train_yolo.py path/to/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --device 0
```

The referenced YAML and its data are private and are not included.

### Train a ResNet Classifier

```bash
python train_resnet.py path/to/classification_data \
  --output weights/resnet18_best.pth
```

The external classification directory must follow the Torchvision
`ImageFolder` layout with separate `train/` and `val/` directories.

### Launch the Interface

```bash
streamlit run app.py
```

The interface asks for a local weight path and a user-provided image.

## Repository Contents

- `app.py`: Streamlit image-detection interface;
- `train_yolo.py`: YOLOv8 training entry point;
- `train_resnet.py`: ResNet18 classification training entry point;
- `tools/remap_labels.py`: generic YOLO class-ID remapping;
- `tools/count_labels.py`: label-frequency summary;
- `tools/frames_to_video.py`: image-sequence to MP4 conversion;
- `environment.yml`: Conda environment specification;
- `requirements.txt`: pip dependencies;
- `.gitignore`: exclusions for data, media, weights, and outputs.

## Weight Release Policy

The local final YOLO checkpoint is approximately 52 MB, and each ResNet
checkpoint is approximately 45 MB. These checkpoints were trained on private
bus-company data and are not included in this snapshot. They should only be
released after explicit authorization from the data owner, together with a model
card and intended-use restrictions.

