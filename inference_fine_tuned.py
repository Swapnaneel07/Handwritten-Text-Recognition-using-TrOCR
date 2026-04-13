import os
import torch
import glob
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the fine-tuned model and processor
checkpoint_path = 'trocr_handwritten/checkpoint-6770'
print(f"\nLoading fine-tuned model from: {checkpoint_path}")

processor = TrOCRProcessor.from_pretrained(checkpoint_path)
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(device)
model.eval()

print("Model loaded successfully!")

def ocr_inference(image_path, processor, model, device):
    """
    Run OCR inference on a single image.
    
    Args:
        image_path: Path to the input image
        processor: TrOCR processor
        model: Fine-tuned TrOCR model
        device: Device to run inference on
        
    Returns:
        generated_text: The OCR'd text
    """
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return image, generated_text


def visualize_results(image_paths, num_samples=5, save_dir='inference_results'):
    """
    Run inference on multiple images and visualize results.
    
    Args:
        image_paths: List of image paths
        num_samples: Number of samples to process
        save_dir: Directory to save result images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_to_process = min(num_samples, len(image_paths))
    
    fig, axes = plt.subplots(num_to_process, 1, figsize=(10, 4 * num_to_process))
    
    if num_to_process == 1:
        axes = [axes]
    
    print(f"\nRunning inference on {num_to_process} images...\n")
    
    for idx, image_path in enumerate(image_paths[:num_to_process]):
        print(f"Processing: {os.path.basename(image_path)}")
        image, text = ocr_inference(image_path, processor, model, device)
        
        axes[idx].imshow(image)
        axes[idx].set_title(f"Predicted: {text}", fontsize=12, wrap=True)
        axes[idx].axis('off')
        
        print(f"  → Predicted text: {text}\n")
    
    plt.tight_layout()
    result_path = os.path.join(save_dir, 'inference_results.png')
    plt.savefig(result_path, dpi=100, bbox_inches='tight')
    print(f"\nResults saved to: {result_path}")
    plt.show()


if __name__ == '__main__':
    # Get test images
    test_image_dir = 'input/gnhk_dataset/test_processed/images/'
    image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
    
    if not image_paths:
        print(f"No images found in {test_image_dir}")
    else:
        print(f"\nFound {len(image_paths)} test images")
        
        # Run inference and visualize
        visualize_results(image_paths, num_samples=5)
