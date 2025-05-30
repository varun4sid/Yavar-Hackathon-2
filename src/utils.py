from sentence_transformers import SentenceTransformer, util
from PIL import Image,ImageDraw

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_metadata(metadata_path):
    metadata = {
        "section_header": "",
        "above_text": "",
        "caption": "",
        "footnote": "",
        "below_text": ""
    }

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip() if value.strip().lower() != "null" else ""

    return metadata

def compute_confidence(prompt, caption):
    embeddings = similarity_model.encode([prompt, caption], convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])
    return round(score, 3)
def parse_metadata_txt(content):
    lines = content.strip().split("\n")
    meta = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip() if value.strip() else None
    return meta


def annotate_image_with_space(image_path, concise, detailed, c_score, d_score):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    padding = 100
    new_img = Image.new("RGB", (width, height + padding), "white")
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.text((10, height + 5), f"[{c_score:.2f}] {concise}", fill="blue")
    draw.text((10, height + 50), f"[{d_score:.2f}] {detailed}", fill="red")

    return new_img