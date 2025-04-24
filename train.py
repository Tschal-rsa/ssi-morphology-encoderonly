import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import math
import torch.nn as nn
import pandas as pd
import Levenshtein  # Add Levenshtein distance calculation

from dataset import SyriacDataset
from model import TransformerClassifier

def decode_sequence(indices, dataset):
    """Decode index sequence to Syriac characters"""
    return ''.join([dataset.idx_to_char[idx.item()] for idx in indices if idx.item() != dataset.input_pad_index])

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50):
    best_val_loss = float('inf')
    max_grad_norm = 1.0
    
    # Read patterns.csv
    patterns_df = pd.read_csv('patterns.csv')
    
    # Open file for saving results (append mode)
    with open('MSS.txt', 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("New Training Session\n")
        f.write("="*50 + "\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_total = 0
        train_zero_correct = 0
        train_nonzero_correct = 0
        train_nonzero_exact = 0
        train_zero_total = 0
        train_nonzero_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].long().to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, mask)
            
            batch_size, seq_len, num_classes = outputs.size()
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            # TODO: padding? loss mask?
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Detach loss before converting to scalar
            train_loss += loss.detach().item()
            predictions = outputs.argmax(dim=-1)
            
            # Calculate various metrics
            zero_mask = (labels == 0)
            nonzero_mask = (labels > 0)
            
            # Zero to zero accuracy
            train_zero_correct += ((predictions == 0) & zero_mask).sum().item()
            train_zero_total += zero_mask.sum().item()
            
            # Non-zero to non-zero accuracy
            train_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
            train_nonzero_total += nonzero_mask.sum().item()
            
            # Non-zero exact match accuracy
            train_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
            
            train_total += (zero_mask | nonzero_mask).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_zero_acc = train_zero_correct / train_zero_total if train_zero_total > 0 else 0
        train_nonzero_acc = train_nonzero_correct / train_nonzero_total if train_nonzero_total > 0 else 0
        train_nonzero_exact_acc = train_nonzero_exact / train_nonzero_total if train_nonzero_total > 0 else 0
        train_overall_acc = (train_zero_correct + train_nonzero_exact) / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_total = 0
        val_zero_correct = 0
        val_nonzero_correct = 0
        val_nonzero_exact = 0
        val_zero_total = 0
        val_nonzero_total = 0
        
        print(f"\n{'='*50}")
        print("Validation Phase")
        print(f"{'='*50}")
        
        # Set random seed based on epoch number
        torch.manual_seed(epoch)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].long().to(device)
                mask = batch['mask'].to(device)
                
                outputs = model(input_ids, mask)
                
                batch_size, seq_len, num_classes = outputs.size()
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                
                # TODO: padding? loss mask?
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                
                # Calculate various metrics
                zero_mask = (labels == 0)
                nonzero_mask = (labels > 0)
                
                # Zero to zero accuracy
                val_zero_correct += ((predictions == 0) & zero_mask).sum().item()
                val_zero_total += zero_mask.sum().item()
                
                # Non-zero to non-zero accuracy
                val_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
                val_nonzero_total += nonzero_mask.sum().item()
                
                # Non-zero exact match accuracy
                val_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
                
                val_total += (zero_mask | nonzero_mask).sum().item()
                
                # Show only one random validation sample
                if batch_idx == 0:
                    print(f"\nValidation Sample (Epoch {epoch+1}):")
                    # Use epoch number to select a different sample each time
                    sample_idx = epoch % batch_size
                    input_seq = input_ids[sample_idx]
                    label_seq = labels[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                    pred_seq = predictions[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                    
                    input_text = decode_sequence(input_seq, val_loader.dataset.dataset)
                    label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                    pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                    
                    label_line, pred_line = format_aligned_output(input_text, label_symbols, pred_symbols)
                    
                    print(f"Input: {input_text}")
                    print(f"Label: {label_line}")
                    print(f"Pred:  {pred_line}")
                    print(f"Loss: {loss.item():.4f}")
                    
                    # Save sample to file
                    with open('MSS.txt', 'a') as f:
                        f.write(f"\nEpoch {epoch+1} Sample:\n")
                        f.write(f"Input: {input_text}\n")
                        f.write(f"Label: {label_line}\n")
                        f.write(f"Pred:  {pred_line}\n")
                        f.write(f"Loss: {loss.item():.4f}\n")
        
        val_loss = val_loss / len(val_loader)
        val_zero_acc = val_zero_correct / val_zero_total if val_zero_total > 0 else 0
        val_nonzero_acc = val_nonzero_correct / val_nonzero_total if val_nonzero_total > 0 else 0
        val_nonzero_exact_acc = val_nonzero_exact / val_nonzero_total if val_nonzero_total > 0 else 0
        val_overall_acc = (val_zero_correct + val_nonzero_exact) / val_total
        
        # Calculate zero and non-zero ratios
        train_zero_ratio = train_zero_total / train_total
        train_nonzero_ratio = train_nonzero_total / train_total
        val_zero_ratio = val_zero_total / val_total
        val_nonzero_ratio = val_nonzero_total / val_total
        
        # Save results to file
        with open('MSS.txt', 'a') as f:
            f.write(f"\nEpoch {epoch+1} Results:\n")
            f.write(f"Training Metrics:\n")
            f.write(f"1. Zero/Non-zero Ratio: {train_zero_ratio:.4f}/{train_nonzero_ratio:.4f}\n")
            f.write(f"2. Zero to Zero Accuracy: {train_zero_acc:.4f}\n")
            f.write(f"3. Non-zero to Non-zero Accuracy: {train_nonzero_acc:.4f}\n")
            f.write(f"4. Non-zero Exact Match Accuracy: {train_nonzero_exact_acc:.4f}\n")
            f.write(f"5. Overall Accuracy: {train_overall_acc:.4f}\n")
            f.write(f"6. Loss: {train_loss:.4f}\n")
            
            f.write(f"\nValidation Metrics:\n")
            f.write(f"1. Zero/Non-zero Ratio: {val_zero_ratio:.4f}/{val_nonzero_ratio:.4f}\n")
            f.write(f"2. Zero to Zero Accuracy: {val_zero_acc:.4f}\n")
            f.write(f"3. Non-zero to Non-zero Accuracy: {val_nonzero_acc:.4f}\n")
            f.write(f"4. Non-zero Exact Match Accuracy: {val_nonzero_exact_acc:.4f}\n")
            f.write(f"5. Overall Accuracy: {val_overall_acc:.4f}\n")
            f.write(f"6. Loss: {val_loss:.4f}\n")
            f.write("="*50 + "\n")
        
        print(f"\n{'='*50}")
        print("\nEpoch Results:")
        print(f"Training Metrics:")
        print(f"1. Zero/Non-zero Ratio: {train_zero_ratio:.4f}/{train_nonzero_ratio:.4f}")
        print(f"2. Zero to Zero Accuracy: {train_zero_acc:.4f}")
        print(f"3. Non-zero to Non-zero Accuracy: {train_nonzero_acc:.4f}")
        print(f"4. Non-zero Exact Match Accuracy: {train_nonzero_exact_acc:.4f}")
        print(f"5. Overall Accuracy: {train_overall_acc:.4f}")
        print(f"6. Loss: {train_loss:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"1. Zero/Non-zero Ratio: {val_zero_ratio:.4f}/{val_nonzero_ratio:.4f}")
        print(f"2. Zero to Zero Accuracy: {val_zero_acc:.4f}")
        print(f"3. Non-zero to Non-zero Accuracy: {val_nonzero_acc:.4f}")
        print(f"4. Non-zero Exact Match Accuracy: {val_nonzero_exact_acc:.4f}")
        print(f"5. Overall Accuracy: {val_overall_acc:.4f}")
        print(f"6. Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
            
            # Save best model info to file
            with open('MSS.txt', 'a') as f:
                f.write(f"\nNew Best Model at Epoch {epoch+1}:\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write("="*50 + "\n")

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into training, validation and test sets
    
    Args:
        dataset: Complete dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate size of each set
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split statistics:")
    print(f"Total size: {total_size}")
    print(f"Training set: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

class CustomLoss(nn.Module):
    def __init__(self, zero_mistake_weight=2.5):
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        self.zero_mistake_weight = zero_mistake_weight
        
    def forward(self, outputs, labels):
        # Calculate base cross entropy loss
        base_loss = self.base_criterion(outputs, labels)  # [batch_size * seq_len]
        
        # Get predicted classes
        predictions = outputs.argmax(dim=-1)  # [batch_size * seq_len]
        
        # Create penalty weights
        # Increase penalty when true label is non-zero but prediction is zero
        weights = torch.ones_like(base_loss)
        zero_mistakes = (predictions == 0) & (labels > 0)
        weights[zero_mistakes] = self.zero_mistake_weight
        
        weighted_loss = (base_loss * weights).mean()
        return weighted_loss

def label_to_symbol(label, patterns_df, label_pad_index=-100):
    """Convert numeric label to corresponding symbol form"""
    if label == 0 or label == label_pad_index:
        return ''
    try:
        symbol = patterns_df[patterns_df['编号'] == label]['符号形式'].values[0]
        return symbol
    except:
        return str(label)

def format_aligned_output(input_text, label_symbols, pred_symbols):
    """Align input text and annotations for display"""
    # Create aligned output
    label_line = ''.join([f"{char}{label}" for char, label in zip(input_text, label_symbols)])
    pred_line = ''.join([f"{char}{pred}" for char, pred in zip(input_text, pred_symbols)])
    
    return label_line, pred_line

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model performance on test set"""
    model.eval()
    test_loss = 0
    test_total = 0
    test_zero_correct = 0
    test_nonzero_correct = 0
    test_nonzero_exact = 0
    test_zero_total = 0
    test_nonzero_total = 0
    test_levenshtein_distance = 0
    
    # Read patterns.csv
    patterns_df = pd.read_csv('patterns.csv')
    
    print(f"\n{'='*50}")
    print("Testing Phase")
    print(f"{'='*50}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].long().to(device)
            mask = batch['mask'].to(device)
            
            outputs = model(input_ids, mask)
            
            batch_size, seq_len, num_classes = outputs.size()
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            
            # Calculate various metrics
            zero_mask = (labels == 0)
            nonzero_mask = (labels > 0)
            
            # Zero to zero accuracy
            test_zero_correct += ((predictions == 0) & zero_mask).sum().item()
            test_zero_total += zero_mask.sum().item()
            
            # Non-zero to non-zero accuracy
            test_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
            test_nonzero_total += nonzero_mask.sum().item()
            
            # Non-zero exact match accuracy
            test_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
            
            # Calculate Levenshtein distance for each sequence in the batch
            for i in range(batch_size):
                start_idx = i * seq_len
                end_idx = (i + 1) * seq_len
                pred_seq = predictions[start_idx:end_idx]
                label_seq = labels[start_idx:end_idx]
                
                # Convert to symbol form
                pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                
                # Calculate Levenshtein distance
                test_levenshtein_distance += Levenshtein.distance(''.join(pred_symbols), ''.join(label_symbols))
            
            test_total += (zero_mask | nonzero_mask).sum().item()
            
            # Show one test sample
            if batch_idx == 0:
                print(f"\nTest Sample:")
                sample_idx = 0
                input_seq = input_ids[sample_idx]
                label_seq = labels[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                pred_seq = predictions[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                
                input_text = decode_sequence(input_seq, test_loader.dataset.dataset)
                label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                
                label_line, pred_line = format_aligned_output(input_text, label_symbols, pred_symbols)
                
                print(f"Input: {input_text}")
                print(f"Label: {label_line}")
                print(f"Pred:  {pred_line}")
                print(f"Loss: {loss.item():.4f}")
    
    test_loss = test_loss / len(test_loader)
    test_zero_acc = test_zero_correct / test_zero_total if test_zero_total > 0 else 0
    test_nonzero_acc = test_nonzero_correct / test_nonzero_total if test_nonzero_total > 0 else 0
    test_nonzero_exact_acc = test_nonzero_exact / test_nonzero_total if test_nonzero_total > 0 else 0
    test_overall_acc = (test_zero_correct + test_nonzero_exact) / test_total
    test_levenshtein_distance = test_levenshtein_distance / test_total  # Normalize by total number of sequences
    
    # Calculate zero and non-zero ratios
    test_zero_ratio = test_zero_total / test_total
    test_nonzero_ratio = test_nonzero_total / test_total
    
    # Save test results to file
    with open('MSS.txt', 'a') as f:
        f.write("\nFinal Test Results:\n")
        f.write(f"1. Zero/Non-zero Ratio: {test_zero_ratio:.4f}/{test_nonzero_ratio:.4f}\n")
        f.write(f"2. Zero to Zero Accuracy: {test_zero_acc:.4f}\n")
        f.write(f"3. Non-zero to Non-zero Accuracy: {test_nonzero_acc:.4f}\n")
        f.write(f"4. Non-zero Exact Match Accuracy: {test_nonzero_exact_acc:.4f}\n")
        f.write(f"5. Overall Accuracy: {test_overall_acc:.4f}\n")
        f.write(f"6. Loss: {test_loss:.4f}\n")
        f.write(f"7. Average Levenshtein Distance: {test_levenshtein_distance:.4f}\n")
    
    print("\nTest Results:")
    print(f"1. Zero/Non-zero Ratio: {test_zero_ratio:.4f}/{test_nonzero_ratio:.4f}")
    print(f"2. Zero to Zero Accuracy: {test_zero_acc:.4f}")
    print(f"3. Non-zero to Non-zero Accuracy: {test_nonzero_acc:.4f}")
    print(f"4. Non-zero Exact Match Accuracy: {test_nonzero_exact_acc:.4f}")
    print(f"5. Overall Accuracy: {test_overall_acc:.4f}")
    print(f"6. Loss: {test_loss:.4f}")
    print(f"7. Average Levenshtein Distance: {test_levenshtein_distance:.4f}")
    
    return test_loss, test_zero_acc, test_nonzero_acc, test_nonzero_exact_acc, test_overall_acc, test_levenshtein_distance

def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SyriacDataset('training.csv', max_length=128)
    print(f"Dataset size: {len(dataset)} sentences")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return
    
    vocab_size = len(dataset.char_to_idx) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Count all occurring classes
    all_labels = set()
    for _, labels in dataset.sentences:
        all_labels.update(labels)
    
    # Get maximum class number
    max_label = max(all_labels)
    num_classes = max_label + 1
    
    print("\nClass Statistics:")
    print(f"Number of unique classes: {len(all_labels)}")
    print(f"Maximum class value: {max_label}")
    print(f"Total number of classes: {num_classes}")
    
    # Use new split function
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = TransformerClassifier(vocab_size=vocab_size, num_classes=num_classes).to(device)
    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    criterion = CustomLoss(zero_mistake_weight=2.0).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate model on test set
    print("\nStarting model evaluation on test set...")
    evaluate_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    main() 