# Yavar Hackathon Assessment-2
*This repository is a submission for the Assessment Problem Statement-2 of the Yavar Hackathon by,* <br>
*Roll No : 22PD40* <br>
*Name : Varun Sithaarth KK*

## Installation and Setup
1. Clone the repository and enter its directory
```bash
git clone https://github.com/varun4sid/image-captioning-model.git
cd image-captioning-model
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

For fine-tuning, I have used the <a href="https://www.kaggle.com/datasets/nikhil7280/coco-image-caption?resource=download">COCO Image Captioning Dataset</a>

I have used the cosine similarity measurement for evaluating the genarated caption against the metadata.