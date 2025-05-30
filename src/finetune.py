"""
This script is used to fine-tune the blip-2 model using COCO dataset
"""
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_path, processor, max_samples=None):
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        self.image_dir = image_dir
        self.processor = processor
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}
        self.entries = []
        for ann in annotations["annotations"]:
            image_file = self.image_id_to_filename[ann["image_id"]]
            caption = ann["caption"]
            self.entries.append((image_file, caption))

        if max_samples is not None:
            self.entries = self.entries[:max_samples]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_file, caption = self.entries[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, text=caption, padding="max_length",
                                truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids']
        return inputs

if __name__ == "__main__":
    image_dir = "../dataset/train2014/train2014"
    annotation_path = "../dataset/annotations_trainval2014/annotations/captions_train2014.json"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cpu")
    model.to(device)

    dataset = COCODataset(image_dir, annotation_path, processor, max_samples=2000)

    training_args = TrainingArguments(
        output_dir="./blip-finetuned-coco_2000sample",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=False,  
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained("./blip-finetuned-coco")
    processor.save_pretrained("./blip-finetuned-coco")

    print("\n✅ Fine-tuning complete. Model saved to ./blip-finetuned-coco")