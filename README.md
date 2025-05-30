# Yavar Hackathon Assessment-2
*This repository is a submission for the Assessment Problem Statement-2 of the Yavar Hackathon by,* <br>
*Roll No : 22PD40* <br>
*Name : Varun Sithaarth KK*

## 🔍 Problem Statement  
**Image Captioning from Contextual Metadata Using Vision-Language Models (VLMs)**  

This project focuses on generating both **concise** and **detailed** image captions using:
- Visual content of the image  
- Surrounding contextual metadata:
  - section_header  
  - above_text  
  - caption  
  - footnote  
  - below_text  

The model is fine-tuned on a subset of the **COCO Image Captioning Dataset** using the open-source **BLIP-2 (base)** model.

---

## 📁 Project Specifications

| Component     | Description                                                  |
|---------------|--------------------------------------------------------------|
| VLM           | salesforce/blip2 (BLIP-2 Base)                               |
| Dataset       | <a href="https://www.kaggle.com/datasets/nikhil7280/coco-image-caption?resource=download">COCO Image Captioning Dataset</a>(30k subset)                  |
| Fine-Tuning   | Conducted using both visual and textual metadata context     |
| Metric        | Cosine similarity used for semantic confidence scoring       |
| Image Types   | Tables, graphs, charts, layout diagrams, logos, photos, etc. |

---

## 💻 Installation & Setup

Follow these steps to get the project up and running on your local machine:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/varun4sid/Yavar-Hackathon-2.git](https://github.com/varun4sid/Yavar-Hackathon-2.git)
    cd Yavar-Hackathon-2
    ```

2.  **Install system packages (Linux only):**

    ```bash
    sudo apt-get install $(cat packages.txt)
    ```

3.  **Set up Python environment:**

    * **Linux/macOS:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    * **Windows (CMD or PowerShell):**
        ```powershell
        python -m venv .venv
        .venv\Scripts\activate
        ```

4.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the fine-tuned model and place it in `src/`:**

    [Download Model](https://drive.google.com/drive/folders/1FNdlGLn9uRxL5y1ozKcd73gQmgP_iy5S?usp=sharing)

---

## 🧠 Usage

You have two options for generating captions: via a graphical user interface (GUI) or through the command line.

### Option 1: Generate captions via GUI

1.  Navigate to the `src/` directory:
    ```bash
    cd src/
    ```
2.  Run the GUI script:
    ```bash
    python gui_caption_gen.py
    ```
3.  In the GUI:
    * Upload an image.
    * Fill in the contextual metadata fields.
    * Click "**Submit & Generate Captions**."
  ![Screenshot from 2025-05-30 23-04-48](https://github.com/user-attachments/assets/95a0c74d-9d00-4a2b-9f0b-d62f5a68da5e)
  ![Screenshot from 2025-05-31 00-35-46](https://github.com/user-attachments/assets/a659ce88-23fa-490b-9353-814730ae8c07)

  
4.  The annotated image will be saved in `output_folder/`, and the captions (with confidence scores) will be saved in `captions.json`.

### Option 2: Command-line caption generation

Ensure you have the following directory structure set up:

* `img_folder/`: Contains your input images (e.g., charts, tables).
* `metadata_folder/`: Contains `.txt` metadata files for each image.

Then, run the caption generation script:

```bash
cd src/
python caption_gen.py
```
---

**Output:**

* **Concise and detailed captions** will be generated.
* **Annotated images** will be saved in `output_folder/`.
* **Captions** will be saved in `captions.json`.

---

## 📝 Metadata Format

Each `.txt` file in your `metadata_folder/` should adhere to the following structure:
section_header: Optional section title
above_text: Text appearing above the image
caption: Embedded caption in the figure
footnote: Any footnotes for the figure
below_text: Text appearing below the image
picture_id: #/pictures/0
Metadata can be populated manually or conveniently added through the GUI.

---

## 📂 Directory Structure
```bash
├── img_folder/           # Input images
├── metadata_folder/      # Metadata for each image
├── output_folder/        # Annotated images + captions.json
├── src/
│   ├── caption_gen.py
│   ├── gui_caption_gen.py
│   ├── model.py
│   ├── utils.py
│   └── blip-finetuned-coco_full/ # Place the downloaded model here
```
---

* All models and libraries utilized in this project are **open-source**




# 📄 BLIP-2 Fine-Tuning Script

## 📝 Overview

This script fine-tunes the [BLIP-2 base model](https://huggingface.co/Salesforce/blip-image-captioning-base) on the **COCO 2014 Image Captioning** dataset. It uses the Hugging Face `transformers` library and PyTorch `Trainer` API for training.

---

## 🗂️ File Structure
```bash
project/
│
├── script.py # This fine-tuning script
├── dataset/
│ └── train2014/ # Folder containing COCO training images
│ └── annotations_trainval2014/ # Folder with annotation files
│ └── captions_train2014.json # COCO 2014 captions file
└── blip-finetuned-coco_full/ # Output directory after fine-tuning
```


---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- Pillow
- COCO 2014 Dataset (images + annotations)

Install dependencies:

```bash
pip install torch torchvision transformers pillow
```

For Windows users, these commands work similarly. Use PowerShell or Command Prompt with an active Python environment.

## 🧾 Dataset

- COCO 2014 (Common Objects in Context)
- Images: train2014/
- Captions: captions_train2014.json

The script reads all (or limited) image-caption pairs using the max_samples argument.

## 📦 Components

### 1. COCODataset Class
A custom PyTorch Dataset that loads images and captions and processes them for training.
```python
COCODataset(image_dir, annotation_path, processor, max_samples=None)
```
- image_dir: Directory containing training images.
- annotation_path: Path to captions_train2014.json.
- processor: BLIP processor (for text/image preprocessing).
- max_samples: Optional cap on dataset size.

The output is formatted for the Hugging Face Trainer API.

### 2. BLIP Model and Processor
```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### 3. TrainingArguments
Key parameters:
-batch_size: 32
-epochs: 3
-learning_rate: 5e-5
-logging_steps: 10
-save_steps: 500
-output_dir: ./blip-finetuned-coco_full
