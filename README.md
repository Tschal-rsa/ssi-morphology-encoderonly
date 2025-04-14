# SSI Morphology Encoder-Only Model

This project implements a transformer-based encoder-only model for morphological analysis of Syriac text. The model is trained to predict morphological patterns for each character in the input text.

## Model Architecture

### Base Architecture
- **Model Type**: Transformer Encoder
- **Input Processing**: Character-level tokenization
- **Output**: Sequence of morphological pattern labels
- **Number of Classes**: 129 (including empty pattern)

### Model Parameters
- **Encoder Layers**: 6
- **Attention Heads**: 8
- **Hidden Dimension**: 512
- **Feedforward Dimension**: 2048
- **Dropout Rate**: 0.1
- **Positional Encoding**: Learned positional embeddings

### Character Mapping
```python
char_to_idx = {
    '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
    'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
    'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
    'C': 20, 'T': 21, ' ': 22
}
```

## Training (train.py)

### Overview
The `train.py` script handles the complete training process of the model, including:
- Data loading and preprocessing
- Model training and validation
- Performance monitoring
- Model saving

### Training Process
1. **Data Preparation**
   - Loads training data from specified files
   - Splits data into training (80%) and validation (20%) sets
   - Applies character-level tokenization

2. **Training Loop**
   - Uses cross-entropy loss
   - Implements Adam optimizer with learning rate scheduling
   - Applies gradient clipping (max norm: 1.0)
   - Implements early stopping based on validation loss

3. **Performance Metrics**
   - Zero/Non-zero ratio
   - Zero to Zero accuracy
   - Non-zero to Non-zero accuracy
   - Non-zero Exact Match accuracy
   - Overall accuracy
   - Training and validation loss

### Usage
Basic training:
```bash
python train.py
```

With custom parameters:
```bash
python train.py --batch_size 32 --num_epochs 100 --learning_rate 0.0001
```

### Training Options
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 0.0001)
- `--model_path`: Path to save the trained model (default: 'best_model.pth')
- `--data_path`: Path to training data (default: 'isaiah_transcription.txt')

## Prediction (parse.py)

### Overview
The `parse.py` script provides tools for using the trained model to predict morphological patterns in Syriac text.

### Input Format
The input text file should follow this format:
```
Is\t1\t1\tXZW> D>C<J> BR >MWY...
Is\t1\t2\tCM<W CMJ> WYWTJ >R<>...
```
Where:
- First column: Book name
- Second column: Chapter number
- Third column: Verse number
- Fourth column: Text content

### Usage

#### Single Sentence Prediction
```bash
python parse.py --input "XZW> D>C<J> BR >MWY"
```

#### File Processing
Basic usage:
```bash
python parse.py --file input.txt
```
This creates `input_output.txt` with the same format as the input file.

#### Advanced Options
1. Custom model:
```bash
python parse.py --model path/to/model.pth --file input.txt
```

2. Custom output file:
```bash
python parse.py --file input.txt --output custom_output.txt
```

3. Full options:
```bash
python parse.py --model path/to/model.pth --file input.txt --output custom_output.txt
```

### Output Format
The output maintains the same format as the input:
```
Is\t1\t1\t[parsed text]
Is\t1\t2\t[parsed text]
...
```
Each line contains:
- Book name
- Chapter number
- Verse number
- Parsed text with morphological patterns

## Requirements
- Python 3.x
- PyTorch
- pandas
- numpy
- tqdm (for training progress bars)
- sklearn (for metrics calculation)

## File Structure
- `train.py`: Training script
- `parse.py`: Prediction script
- `model.py`: Transformer model implementation
- `dataset.py`: Dataset handling and preprocessing
- `patterns.csv`: Mapping of pattern labels to symbols
- `best_model.pth`: Default trained model weights

## Notes
- The model processes text character by character
- Each character in the output is followed by its predicted morphological pattern
- Training uses a validation split of 20% of the data
- The model implements gradient clipping to prevent exploding gradients
- Early stopping is implemented to prevent overfitting 