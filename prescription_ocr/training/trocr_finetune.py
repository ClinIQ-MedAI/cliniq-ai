"""
TrOCR Fine-tuning Script for Drug Name Recognition
Uses Microsoft's TrOCR model fine-tuned on synthetic drug name images.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

# Check for required packages
try:
    from transformers import (
        TrOCRProcessor, 
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator
    )
    from datasets import Dataset, load_dataset
    from PIL import Image
    import evaluate
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install with: pip install transformers datasets evaluate pillow")
    raise


@dataclass
class TrOCRConfig:
    """Configuration for TrOCR fine-tuning."""
    model_name: str = "microsoft/trocr-base-handwritten"
    output_dir: str = "./trocr_finetuned"
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data parameters
    max_length: int = 32  # Max text length
    image_size: tuple = (384, 384)
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100


class DrugNameDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for drug name images."""
    
    def __init__(
        self, 
        data: list, 
        processor: TrOCRProcessor,
        max_length: int = 32
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        if isinstance(item.get("image"), Image.Image):
            image = item["image"]
        else:
            image = Image.open(item["image_path"]).convert("RGB")
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Process text label
        labels = self.processor.tokenizer(
            item["label"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


class TrOCRTrainer:
    """Handles TrOCR fine-tuning for drug name recognition."""
    
    def __init__(self, config: TrOCRConfig = None):
        self.config = config or TrOCRConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[TrOCR] Using device: {self.device}")
        
        # Load processor and model
        print(f"[TrOCR] Loading model: {self.config.model_name}")
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        
        # Configure model for training
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        print("[TrOCR] ✓ Model loaded successfully")
    
    def load_dataset(self, labels_path: str) -> Dict[str, DrugNameDataset]:
        """Load training dataset from labels JSON file."""
        print(f"[TrOCR] Loading dataset from {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Split data into train/val (90/10)
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"[TrOCR] Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        return {
            "train": DrugNameDataset(train_data, self.processor, self.config.max_length),
            "val": DrugNameDataset(val_data, self.processor, self.config.max_length)
        }
    
    def compute_metrics(self, pred):
        """Compute Character Error Rate (CER) metric."""
        import numpy as np
        cer_metric = evaluate.load("cer")
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Handle numpy array conversion
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        # Convert to numpy if needed
        if not isinstance(pred_ids, np.ndarray):
            pred_ids = np.array(pred_ids)
        if not isinstance(label_ids, np.ndarray):
            label_ids = np.array(label_ids)
        
        # Clip prediction IDs to valid vocabulary range to avoid overflow errors
        vocab_size = self.processor.tokenizer.vocab_size
        pred_ids = np.clip(pred_ids, 0, vocab_size - 1).astype(np.int32)
        
        # Replace -100 with pad token id (make a copy first)
        label_ids = np.where(label_ids == -100, self.processor.tokenizer.pad_token_id, label_ids)
        label_ids = label_ids.astype(np.int32)
        
        # Decode predictions and labels
        try:
            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"[Warning] Decoding error: {e}")
            return {"cer": 1.0}  # Return worst case if decoding fails
        
        # Compute CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"cer": cer}
    
    def train(self, train_dataset, val_dataset):
        """Fine-tune the TrOCR model."""
        print("[TrOCR] Starting training...")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            eval_strategy="steps",
            save_strategy="steps",  # Must match eval_strategy for load_best_model_at_end
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,  # Lower CER is better
            push_to_hub=False,
            dataloader_num_workers=4,  # Speed up data loading
            report_to="none",  # Disable wandb/tensorboard if not set up
            save_total_limit=2,  # Keep only 2 checkpoints to save disk space
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor.tokenizer,  # Use processing_class instead of deprecated tokenizer
            data_collator=default_data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save the final model
        self.model.save_pretrained(self.config.output_dir)
        self.processor.save_pretrained(self.config.output_dir)
        
        print(f"[TrOCR] ✓ Model saved to {self.config.output_dir}")
        
        return trainer
    
    def predict(self, image_path: str) -> str:
        """Make a prediction on a single image."""
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        self.model.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        
        # Decode
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    
    @classmethod
    def load_trained_model(cls, model_dir: str) -> "TrOCRTrainer":
        """Load a trained model from disk."""
        config = TrOCRConfig(model_name=model_dir, output_dir=model_dir)
        trainer = cls(config)
        return trainer


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for drug names")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.json")
    parser.add_argument("--output", type=str, default="./models/trocr_drugs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Configure and train
    config = TrOCRConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    trainer = TrOCRTrainer(config)
    datasets = trainer.load_dataset(args.labels)
    trainer.train(datasets["train"], datasets["val"])
    
    print("\n[Complete] ✓ TrOCR fine-tuning finished!")


if __name__ == "__main__":
    main()
