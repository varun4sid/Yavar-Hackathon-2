# Yavar Hackathon Assessment-2
*This repository is a submission for the Assessment Problem Statement-2 of the Yavar Hackathon by,* <br>
*Roll No : 22PD40* <br>
*Name : Varun Sithaarth KK*

## Installation and Setup
1. Clone the repository and enter its directory
```bash
git clone https://github.com/varun4sid/Yavar-Hackathon-2.git
cd Yavar-Hackathon-2
```
2. Install the necessary system packages (If you're on Windows, make sure to have python3, pip and python3-venv installed, and skip this step)
```bash
sudo apt-get install $(cat packages.txt)
```
3. Create a virtual environment and activate it.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
4. Install the necessary libraries in the environment.
```bash
pip install -r requirements.txt
```
5. Download the fine-tuned model uploaded <a href="https://drive.google.com/file/d/1QT0MqO-rCJaonfdYyB3l1gUUVrN8Ab7q/view?usp=drive_link">here</a> and place it in src/

## Project Specifications
<table>
<tbody>
<tr><td>VLM</td><td>BLIP-2-base</td></tr>
<tr><td>Dataset</td><td><a href="https://www.kaggle.com/datasets/nikhil7280/coco-image-caption?resource=download">COCO Image Captioning Dataset</a></td></tr>
</tbody>
</table>
- I have used the cosine similarity measurement for evaluating the genarated caption.

## Usage
1. Add image and context through a GUI.
```bash
cd src/
python3 gui_caption_gen.py
```
![Screenshot from 2025-05-30 23-04-48](https://github.com/user-attachments/assets/95a0c74d-9d00-4a2b-9f0b-d62f5a68da5e)

2. Run the following command that generates output in the output folder
```bash
python3 caption_gen.py
```

## Challenges Faced
+ I couldn't find a dataset with images of tables, charts, etc., along with the metadata context for fine tuning.
+ I couldn't utilize my GPU as the CUDA drivers required by Pytorch couldn't communicate with my GPU.
+ As a result, fine-tuning using the COCO dataset took 8+ hours using the CPU.
