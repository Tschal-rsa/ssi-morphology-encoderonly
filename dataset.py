import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SyriacDataset(Dataset):
    def __init__(self, csv_file, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        
        # Create character to index mapping
        self.char_to_idx = {
            '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
            'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
            'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
            'C': 20, 'T': 21, ' ': 22
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Process data, group by sentences
        self.sentences = []
        current_sentence = []
        current_labels = []
        
        def split_sentence(sentence, labels):
            """Intelligently split sentence, preferably at spaces"""
            if len(sentence) <= self.max_length:
                return [(sentence, labels)]
            
            segments = []
            start = 0
            
            while start < len(sentence):
                # Find next split point (last space within max_length)
                end = min(start + self.max_length, len(sentence))
                if end < len(sentence):
                    # Look for last space from end backwards within max_length
                    for i in range(end, start, -1):
                        if sentence[i] == ' ' and sentence[i-1] == ' ':
                            end = i
                            break
                
                # If no suitable space found, force split at max_length
                if end == start + self.max_length:
                    # Look for nearest single space
                    for i in range(end, start, -1):
                        if sentence[i] == ' ':
                            end = i
                            break
                
                segments.append((sentence[start:end], labels[start:end]))
                start = end
            
            return segments
        
        for _, row in self.data.iterrows():
            if pd.isna(row['input']) or (row['input'] == ' ' and len(current_sentence) > 0 and current_sentence[-1] == ' '):
                if current_sentence:
                    # Use intelligent split function to process current sentence
                    segments = split_sentence(current_sentence, current_labels)
                    self.sentences.extend(segments)
                    current_sentence = []
                    current_labels = []
            else:
                current_sentence.append(row['input'])
                # Modify label processing logic
                output_value = row['output']
                if isinstance(output_value, str):
                    # If output contains multiple values, take the first one
                    output_value = output_value.split(',')[0]
                output_value = int(output_value)  # Convert string to integer
                current_labels.append(output_value)  # Use raw value directly, no restrictions
        
        if current_sentence:
            # Process last sentence
            segments = split_sentence(current_sentence, current_labels)
            self.sentences.extend(segments)
        
        print(f"Total sentences found: {len(self.sentences)}")
        
        # Count sequence lengths
        lengths = [len(sentence) for sentence, _ in self.sentences]
        print(f"\nSequence length statistics:")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Median length: {np.median(lengths):.2f}")
        print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
        
        if len(self.sentences) > 0:
            print(f"\nSample sentence: {''.join(self.sentences[0][0])}")
            print(f"Sample labels: {self.sentences[0][1]}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence, labels = self.sentences[idx]
        
        # Convert characters to indices
        sentence_indices = [self.char_to_idx[char] for char in sentence]
        
        # Add padding, but only up to the actual needed length
        if len(sentence_indices) < self.max_length:
            sentence_indices.extend([self.char_to_idx[' ']] * (self.max_length - len(sentence_indices)))
        else:
            sentence_indices = sentence_indices[:self.max_length]
        
        # Ensure label length matches sentence length
        if len(labels) < self.max_length:
            labels.extend([0] * (self.max_length - len(labels)))
        else:
            labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(sentence_indices, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        } 