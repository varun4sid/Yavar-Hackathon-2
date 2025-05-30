from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = torch.device("cpu")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned-coco").to(device)
processor = BlipProcessor.from_pretrained("./blip-finetuned-coco")
model.eval()

def generate_captions(image_path, metadata):
    image = Image.open(image_path).convert("RGB")

    if any(metadata.values()):
        concise_prompt = f"Give a short summary of this figure. {metadata.get('section_header', '')}"
        context = " ".join([
            metadata.get("section_header", ""),
            metadata.get("above_text", ""),
            metadata.get("caption", ""),
            metadata.get("footnote", ""),
            metadata.get("below_text", "")
        ])
        detailed_prompt = f"Describe this figure in detail considering the context: {context}"

        with torch.no_grad():
            concise_inputs = processor(image, concise_prompt, return_tensors="pt").to(device)
            concise_output = model.generate(**concise_inputs, max_length=64)
            concise_caption = processor.decode(concise_output[0], skip_special_tokens=True)

            detailed_inputs = processor(image, detailed_prompt, return_tensors="pt").to(device)
            detailed_output = model.generate(**detailed_inputs, max_length=128)
            detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)

        return concise_caption, detailed_caption, concise_prompt, detailed_prompt

    else:
        with torch.no_grad():
            inputs = processor(image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_length=64)
            caption = processor.decode(output[0], skip_special_tokens=True)

        return caption, caption, "", ""