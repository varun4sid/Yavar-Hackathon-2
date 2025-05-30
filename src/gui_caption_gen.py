"""
A GUI for uploading images and context to get a caption-annotated image from the fine-tuned model
"""
import os
import json
from tkinter import Tk, filedialog, Label, Button, Entry, StringVar
from PIL import Image, ImageTk
from model import generate_captions
from utils import compute_confidence, annotate_image_with_space

IMG_FOLDER = "../img_folder"
META_FOLDER = "../metadata_folder"
OUTPUT_FOLDER = "../output_folder"
CAPTIONS_JSON = os.path.join(OUTPUT_FOLDER, "captions.json")

os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(META_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

root = Tk()
root.title("Contextual Image Captioning")
root.geometry("700x700")

img_path = ""
img_tk = None

metadata_vars = {
    "section_header": StringVar(),
    "above_text": StringVar(),
    "caption": StringVar(),
    "footnote": StringVar(),
    "below_text": StringVar()
}


def save_files_and_generate():
    global img_path

    if not img_path:
        caption_label.config(text="⚠️ Please select an image.")
        return

    img_filename = os.path.basename(img_path)
    meta_filename = os.path.splitext(img_filename)[0] + ".txt"

    # Save image
    saved_img_path = os.path.join(IMG_FOLDER, img_filename)
    Image.open(img_path).save(saved_img_path)

    # Save metadata
    metadata = {key: var.get().strip() for key, var in metadata_vars.items()}
    with open(os.path.join(META_FOLDER, meta_filename), "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}:{value}\n")

    # Generate captions
    concise, detailed, cp, dp = generate_captions(saved_img_path, metadata)
    c_score = compute_confidence(concise, metadata)
    d_score = compute_confidence(detailed, metadata)

    # Annotate and save
    annotated = annotate_image_with_space(saved_img_path, concise, detailed, c_score, d_score)
    output_img_path = os.path.join(OUTPUT_FOLDER, img_filename)
    annotated.save(output_img_path)

    if os.path.exists(CAPTIONS_JSON):
        with open(CAPTIONS_JSON, "r") as f:
            try:
                all_captions = json.load(f)
                if not isinstance(all_captions, dict):
                    all_captions = {}
            except json.JSONDecodeError:
                all_captions = {}
    else:
        all_captions = {}


    all_captions[img_filename] = {
        "concise": concise,
        "detailed": detailed,
        "confidence": {
            "concise": round(c_score, 3),
            "detailed": round(d_score, 3)
        },
        "prompts": {
            "concise": cp,
            "detailed": dp
        }
    }

    with open(CAPTIONS_JSON, "w") as f:
        json.dump(all_captions, f, indent=2)

    caption_label.config(text="✅ Captioning Complete. Output saved.")
    
    root.destroy()

def upload_image():
    global img_path, img_tk
    img_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if img_path:
        img = Image.open(img_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

Label(root, text="Upload an Image and Enter Metadata", font=("Arial", 14)).pack(pady=10)

Button(root, text="Upload Image", command=upload_image).pack(pady=5)

# Metadata fields
for field in metadata_vars:
    Label(root, text=field.replace("_", " ").title()).pack()
    Entry(root, textvariable=metadata_vars[field], width=60).pack(pady=2)

image_label = Label(root)
image_label.pack(pady=10)

Button(root, text="Submit & Generate Captions", command=save_files_and_generate).pack(pady=10)

caption_label = Label(root, text="", wraplength=600, font=("Arial", 12))
caption_label.pack(pady=20)

root.mainloop()