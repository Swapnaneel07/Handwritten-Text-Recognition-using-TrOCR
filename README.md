# Handwritten Text Recognition using OCR

## Project Overview

This project implements a fine-tuned Optical Character Recognition (OCR) system specifically designed for handwritten text recognition. It fine-tunes Microsoft's TrOCR (Transformer-based OCR) model on the GoodNotes Handwritten (GNHK) dataset to improve accuracy over the pretrained baseline. The system includes complete pipelines for data preprocessing, model training, inference, and deployment through multiple web interfaces.

### Key Features
- Fine-tuning of TrOCR model for handwritten text
- Data preprocessing from raw page images to individual word crops
- Multiple inference modes: batch processing, Flask web API, Gradio UI
- Performance evaluation using Character Error Rate (CER)
- Achieved CER of ~0.2302 on test set

## Technical Architecture

### Core Components
- **TrOCR Model**: Vision Encoder-Decoder architecture using Vision Transformer (ViT) encoder and sequence-to-sequence decoder
- **Transformers Library**: Hugging Face implementation for model loading and training
- **PyTorch**: Deep learning framework for training and inference
- **OpenCV**: Image processing for data preprocessing

### Model Configuration
- Model: `microsoft/trocr-small-handwritten`
- Max sequence length: 64
- Beam search: 4 beams
- Early stopping enabled
- No repeat n-gram size: 3
- Length penalty: 2.0

## Files and Their Purposes

### Training and Model Development
- **[Fine_Tune_TrOCR_Handwritten.ipynb](Fine_Tune_TrOCR_Handwritten.ipynb)**: Main training pipeline that fine-tunes the TrOCR model on the GNHK dataset. Includes data loading, augmentation, model configuration, training loop, and evaluation.
- **[preprocess_gnhk_dataset.py](preprocess_gnhk_dataset.py)**: Preprocesses raw GNHK dataset by cropping individual words from full page images using JSON annotations and creating CSV files mapping image filenames to text labels.

### Inference and Evaluation
- **[inference_fine_tuned.py](inference_fine_tuned.py)**: Batch inference script that runs OCR on test images and visualizes results in a grid format.
- **[TrOCR_Inference.ipynb](TrOCR_Inference.ipynb)**: Comprehensive inference notebook with device checks, model loading, single/batch processing, and custom image handling.
- **[Pretrained_Model_Inference.ipynb](Pretrained_Model_Inference.ipynb)**: Baseline testing using the unfine-tuned pretrained TrOCR model for comparison.

### Web Applications
- **[trocr_flask_app.py](trocr_flask_app.py)**: Flask web application providing a REST API and HTML interface for uploading images and getting OCR results.
- **[trocr_ui_app.py](trocr_ui_app.py)**: Lightweight Gradio-based web interface for easy OCR inference with auto-generated UI.

### Data and Results
- **input/gnhk_dataset/**: Contains the GNHK dataset with raw images, annotations, and processed data.
  - `train_data/train/` and `test_data/test/`: Raw page images and JSON annotations
  - `train_processed/` and `test_processed/`: Cropped word crops and CSV mappings
- **trocr_handwritten/**: Fine-tuned model checkpoints and training artifacts.
  - `checkpoint-6093/` and `checkpoint-6770/`: Saved model states with best CER at step 6770
  - `runs/`: TensorBoard logs for training monitoring
- **inference_results/**: Output directory for inference results and visualizations.

## Data Pipeline

### Preprocessing Steps
1. **JSON Parsing**: Load annotations with polygon coordinates and text from JSON files
2. **Bounding Box Conversion**: Convert polygons to axis-aligned bounding boxes using OpenCV
3. **Image Cropping**: Extract individual word regions from full page images
4. **Text Normalization**: Handle special characters (e.g., replace `%text%` with `SPECIAL_CHARACTER`)
5. **CSV Generation**: Create mapping files with image filenames and corresponding text labels

### Dataset Structure
- **Input**: Full page handwritten images with JSON annotations
- **Output**: Individual word crops (JPG) and CSV files with filename-text pairs
- **Augmentation**: Color jitter and Gaussian blur applied during training

## Training Process

### Configuration
- **Batch Size**: 48
- **Epochs**: 10
- **Learning Rate**: 5e-5 (AdamW optimizer)
- **Weight Decay**: 0.0005
- **Mixed Precision**: FP16 enabled for faster training
- **Evaluation**: Character Error Rate (CER) computed at each epoch

### Workflow
1. Load pretrained TrOCR model
2. Freeze vision encoder, fine-tune decoder and cross-attention layers
3. Apply data augmentation during training
4. Use Seq2SeqTrainer for end-to-end training
5. Save checkpoints and monitor progress with TensorBoard
6. Select best model based on validation CER

## Inference Mechanisms

### Generation Parameters
- Beam search with 4 beams for improved accuracy
- Maximum length of 64 tokens
- Early stopping when EOS token is generated
- N-gram repetition prevention

### Deployment Options
1. **Batch Processing**: Script-based inference for multiple images
2. **Flask API**: Production-ready REST API with HTML interface
3. **Gradio UI**: Quick deployment with auto-generated responsive interface

## Dependencies

### Core ML Libraries
- `transformers`: Hugging Face library for TrOCR model and tokenization
- `torch` & `torchvision`: PyTorch deep learning framework and image utilities
- `datasets`: Hugging Face datasets library for data handling
- `evaluate`: Library for computing evaluation metrics like CER
- `accelerate`: Multi-GPU training acceleration

### Image Processing
- `Pillow` (PIL): Image loading, conversion, and manipulation
- `opencv-python` (cv2): Computer vision operations for bounding box processing

### Data Science
- `numpy`: Numerical computing and array operations
- `pandas`: Data manipulation and CSV handling
- `matplotlib`: Visualization and plotting

### Web Frameworks
- `flask`: Web framework for REST API development
- `gradio`: Auto-UI generation for machine learning models

### Utilities
- `sentencepiece`: Tokenization library required by TrOCR
- `jiwer`: Text similarity metrics for CER calculation
- `protobuf==3.20.1`: Protocol buffers for TensorBoard compatibility
- `tensorboard`: Training visualization and monitoring

### Development
- `jupyter`: Interactive notebook environment
- `tqdm`: Progress bars for long-running operations

## Usage

### Training
1. Run `preprocess_gnhk_dataset.py` to prepare the dataset
2. Execute `Fine_Tune_TrOCR_Handwritten.ipynb` to fine-tune the model

### Inference
- **Batch**: Run `inference_fine_tuned.py` for multiple images
- **Web API**: Execute `trocr_flask_app.py` and access via browser
- **Gradio UI**: Run `trocr_ui_app.py` for interactive interface

### Evaluation
- Use `TrOCR_Inference.ipynb` for comprehensive testing
- Compare with `Pretrained_Model_Inference.ipynb` for baseline performance

## Performance

- **Metric**: Character Error Rate (CER)
- **Best Result**: CER ≈ 0.2302 (achieved at checkpoint-6770)
- **Interpretation**: Approximately 23% of characters differ from ground truth on average

## Technical Highlights

1. **Vision Transformer Encoder**: Pretrained on ImageNet, provides robust image feature extraction
2. **Sequence-to-Sequence Decoder**: Generates text tokens autoregressively
3. **Mixed Precision Training**: FP16 reduces memory usage and speeds up training
4. **Data Augmentation**: Improves model robustness to variations in handwriting
5. **Beam Search Decoding**: Explores multiple hypotheses for better accuracy
6. **TensorBoard Integration**: Real-time monitoring of training metrics
7. **Multiple Deployment Options**: From lightweight demos to production APIs

This project demonstrates a complete OCR pipeline from data preparation through model deployment, optimized for handwritten text recognition using state-of-the-art transformer architectures.