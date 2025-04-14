import torch
import random
from model import TransformerClassifier
from dataset import SyriacDataset
import re
import pandas as pd

# Load model
def load_model(model_path, device):
    # Set num_classes to 129 to match the saved model
    model = TransformerClassifier(num_classes=129)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Load patterns.csv
def load_patterns(pattern_file):
    patterns_df = pd.read_csv(pattern_file)
    return patterns_df

def label_to_symbol(label, patterns_df):
    """Convert numeric label to corresponding symbol form"""
    if label == 0:
        return ''
    try:
        symbol = patterns_df[patterns_df['编号'] == label]['符号形式'].values[0]
        return symbol
    except:
        return str(label)

def format_aligned_output(input_text, pred_symbols):
    """Align input text and prediction annotations for display"""
    # Ensure input text and prediction annotations have same length
    min_len = min(len(input_text), len(pred_symbols))
    input_text = input_text[:min_len]
    pred_symbols = pred_symbols[:min_len]
    
    # Create aligned output
    pred_line = ''.join([f"{char}{pred}" for char, pred in zip(input_text, pred_symbols)])
    
    return pred_line

# Load text file and split into sentences
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Use regex to split text, preserving chapter markers
    sentences = []
    current_sentence = []
    current_line_number = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Handle chapter markers separately
        if line.startswith('Chapter'):
            if current_sentence:
                sentences.append((''.join(current_sentence), current_line_number))  # Save sentence and line number
                current_sentence = []
                current_line_number = None
            sentences.append((line, None))
        else:
            # Process regular text lines, extract line numbers
            parts = re.split(r'^(\d+)\s+', line)  # Extract numbers at the beginning of line
            if len(parts) > 1:
                current_line_number = parts[1]
                line = parts[2]
            current_sentence.extend(list(line.strip()))  # Convert each line to character list
    
    if current_sentence:
        sentences.append((''.join(current_sentence), current_line_number))
    
    return sentences

# Convert text to model input
def text_to_tensor(text, char_to_idx, max_length=128):
    # Convert characters to indices
    indices = [char_to_idx.get(c, char_to_idx[' ']) for c in text]
    
    # Add padding
    if len(indices) < max_length:
        indices.extend([char_to_idx[' ']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension

def split_sentence(sentence, max_length=50):
    """
    Intelligently split a sentence into segments that don't exceed max_length.
    Each segment will end with two spaces to maintain consistency with training data.
    """
    words = sentence.split()
    segments = []
    current_segment = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed max_length
        if current_length + len(word) + len(current_segment) > max_length and current_segment:
            # Join current segment and add two spaces
            segments.append(' '.join(current_segment) + '  ')
            current_segment = []
            current_length = 0
        
        current_segment.append(word)
        current_length += len(word)
    
    # Add any remaining words
    if current_segment:
        segments.append(' '.join(current_segment) + '  ')
    
    return segments if segments else [sentence + '  ']  # Ensure we return at least one segment

def main(input_sentence=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model('best_model.pth', device)
    
    # Load patterns
    patterns_df = load_patterns('patterns.csv')
    
    # Prepare character mapping
    char_to_idx = {
        '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
        'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
        'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
        'C': 20, 'T': 21, ' ': 22
    }
    
    if input_sentence is None:
        # Load text
        sentences = load_sentences('isaiah_transcription.txt')
        
        # Filter out chapter markers, only select actual text lines
        text_sentences = [s for s in sentences if not s[0].startswith('Chapter')]
        if not text_sentences:
            print("No valid sentences found!")
            return
        
        # Randomly select a sentence
        random_sentence, line_number = random.choice(text_sentences)
        sentence_to_process = random_sentence
        print(f"\nSelected text (Line {line_number}):")
    else:
        sentence_to_process = input_sentence
        line_number = None
        print(f"\nInput text: {sentence_to_process}")
        
        # Process the entire input sentence directly, without segmentation
        sentence_segments = [sentence_to_process]
    
    # Use the same segmentation logic as in training (if not direct input)
    if input_sentence is None:
        sentence_segments = split_sentence(sentence_to_process)
        segment = random.choice(sentence_segments)
        print(f"Processing segment: {segment}")
    else:
        segment = sentence_to_process
        print(f"Processing input: {segment}")
    
    # Convert to model input
    input_tensor = text_to_tensor(segment, char_to_idx)
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=-1)
    
    # Convert predictions to symbol form
    pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in predictions[0]]
    
    # Create aligned output
    pred_line = format_aligned_output(segment, pred_symbols)
    print(f"Pred:  {pred_line}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run best model prediction')
    parser.add_argument('--input', type=str, help='Input sentence to predict')
    args = parser.parse_args()
    main(args.input)

if __name__ == "__main__":
    main()
