# SSI Morphology Encoder-Only

This project is a transformer-based model for Syriac morphology analysis. It uses an encoder-only architecture to predict morphological patterns in Syriac text.

## Requirements

- Python 3.9
- PyTorch
- pandas
- tqdm
- Levenshtein
- sklearn

## Project Structure

- `train.py`: Script for training the model
- `parse.py`: Script for parsing and predicting Syriac text
- `model.py`: Contains the transformer model architecture
- `dataset.py`: Handles data loading and preprocessing
- `patterns.csv`: Contains the mapping between labels and morphological patterns

## Usage

### Training the Model

To train the model, use the `train.py` script:

```bash
python train.py
```

The script will:
1. Load and preprocess the training data
2. Train the transformer model
3. Save the best model to `best_model.pth`
4. Display training metrics including:
   - Zero/Non-zero ratio
   - Zero to Zero accuracy
   - Non-zero to Non-zero accuracy
   - Non-zero Exact Match accuracy
   - Overall accuracy
   - Loss values

### Parsing Text

To parse Syriac text and predict morphological patterns, use the `parse.py` script:

```bash
# Parse a random sentence from the dataset
python parse.py

# Parse a specific input sentence
python parse.py --input "Your Syriac text here"
```

The script will:
1. Load the trained model
2. Process the input text
3. Display the input text with predicted morphological patterns aligned below each character

## Output Format

The output shows the input text with predicted morphological patterns aligned below each character. For example:

```
Input: B>L
Pred:  B>L
      -a-
```

Where:
- The first line shows the input Syriac text
- The second line shows the predicted morphological patterns aligned with each character

## Notes

- The model uses a transformer encoder architecture
- Training data should be in the format of Syriac text with corresponding morphological patterns
- The model supports both training and inference modes
- The `patterns.csv` file contains the mapping between numerical labels and morphological patterns 