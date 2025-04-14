# SSI Morphology Encoder-Only Model

This project implements a transformer-based encoder-only model for morphological analysis of Syriac text. The model is trained to predict morphological patterns for each character in the input text.

## Technical Details

### Model Architecture
- **Base Model**: Transformer Encoder
- **Input Processing**: Character-level tokenization
- **Output**: Sequence of morphological pattern labels
- **Number of Classes**: 129 (including empty pattern)

### Character Mapping
The model uses the following character mapping for input processing:
```python
char_to_idx = {
    '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
    'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
    'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
    'C': 20, 'T': 21, ' ': 22
}
```

### Data Format
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

## Usage

### Basic Usage
Process a single input sentence:
```bash
python parse.py --input "XZW> D>C<J> BR >MWY"
```

### File Processing
Process an entire text file:
```bash
python parse.py --file input.txt
```
This will create an output file named `input_output.txt` with the same format as the input file.

### Advanced Options
1. Specify a custom model file:
```bash
python parse.py --model path/to/model.pth --file input.txt
```

2. Specify a custom output file:
```bash
python parse.py --file input.txt --output custom_output.txt
```

3. Process a file with a different model and custom output:
```bash
python parse.py --model path/to/model.pth --file input.txt --output custom_output.txt
```

### Output Format
The output file maintains the same format as the input file:
```
Is\t1\t1\t[parsed text]
Is\t1\t2\t[parsed text]
...
```
Where each line contains:
- Book name
- Chapter number
- Verse number
- Parsed text with morphological patterns

## File Structure
- `parse.py`: Main script for text parsing
- `model.py`: Transformer model implementation
- `dataset.py`: Dataset handling and preprocessing
- `patterns.csv`: Mapping of pattern labels to symbols
- `best_model.pth`: Default trained model weights

## Requirements
- Python 3.x
- PyTorch
- pandas
- numpy

## Notes
- The model processes text character by character
- Each character in the output is followed by its predicted morphological pattern
- The output file maintains the same structure as the input file for easy comparison 