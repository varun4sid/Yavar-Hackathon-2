"""
This script is used to generate captions for the images in img_folder using context from metadata_folder
and stores the annotated image in output_folder
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from utils import load_metadata, compute_confidence
from PIL import Image, ImageDraw, ImageFont


image_dir = "../img_folder"
metadata_dir = "../metadata_folder"
output_dir = "../output_folder"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cpu")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned-coco").to(device)
processor = BlipProcessor.from_pretrained("./blip-finetuned-coco")
model.eval()

font = ImageFont.load_default()
results = []

def generate_captions(image_path, metadata):
    image = Image.open(image_path).convert("RGB")

    # Check if any metadata field is non-empty
    metadata_text = " ".join([
        metadata.get("section_header", "").strip(),
        metadata.get("above_text", "").strip(),
        metadata.get("caption", "").strip(),
        metadata.get("footnote", "").strip(),
        metadata.get("below_text", "").strip()
    ])
    use_prompts = bool(metadata_text.strip())

    concise_prompt = None
    detailed_prompt = None

    with torch.no_grad():
        if use_prompts:
            concise_prompt = f"Give a short summary of this figure. {metadata.get('section_header', '')}"
            context = " ".join([
                metadata.get("section_header", ""),
                metadata.get("above_text", ""),
                metadata.get("caption", ""),
                metadata.get("footnote", ""),
                metadata.get("below_text", "")
            ])
            detailed_prompt = f"Describe this figure in detail considering the context: {context}"

            concise_inputs = processor(image, concise_prompt, return_tensors="pt").to(device)
            concise_output = model.generate(**concise_inputs, max_length=64)
            concise_caption = processor.decode(concise_output[0], skip_special_tokens=True)

            detailed_inputs = processor(image, detailed_prompt, return_tensors="pt").to(device)
            detailed_output = model.generate(**detailed_inputs, max_length=128)
            detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)
        else:
            inputs = processor(image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_length=64)
            caption = processor.decode(output[0], skip_special_tokens=True)
            concise_caption = detailed_caption = caption

    return concise_caption, detailed_caption, concise_prompt, detailed_prompt


def annotate_image(image_path, concise, detailed, c_score, d_score):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    try:
        font = ImageFont.truetype("arialbd.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    bar_height = height // 4
    new_height = height + bar_height
    new_img = Image.new("RGB", (width, new_height), color="white")
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    padding = 10
    draw.text((padding, height + padding), f"[{c_score:.2f}] {concise}", fill="blue", font=font)
    draw.text((padding, height + padding + 30), f"[{d_score:.2f}] {detailed}", fill="red", font=font)

    return new_img


for file in os.listdir(image_dir):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, file)
    metadata_path = os.path.join(metadata_dir, os.path.splitext(file)[0] + ".txt")

    if not os.path.exists(metadata_path):
        print(f"❌ Missing metadata for {file}")
        continue

    metadata = load_metadata(metadata_path)
    concise, detailed, c_prompt, d_prompt = generate_captions(image_path, metadata)
    c_score = compute_confidence(c_prompt, concise)
    d_score = compute_confidence(d_prompt, detailed)

    annotated = annotate_image(image_path, concise, detailed, c_score, d_score)
    annotated.save(os.path.join(output_dir, file))

    results.append({
        "image": file,
        "concise_caption": concise,
        "concise_confidence": c_score,
        "detailed_caption": detailed,
        "detailed_confidence": d_score
    })

with open(os.path.join(output_dir, "captions.json"), "w") as f:
    json.dump(results, f, indent=4)

print("✅ Caption generation complete.")