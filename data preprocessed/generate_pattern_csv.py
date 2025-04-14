import re
from collections import Counter
import csv

def is_syriac_letter(char):
    """Check if a character is a Syriac letter (defined as A-Z, < or >)"""
    return char.isalpha() and char.isupper() or char in ['<', '>']

def analyze_symbols_between_letters(text):
    """
    Analyze all symbol patterns between Syriac letters within words,
    as well as before the first letter and after the last letter.
    Returns a dictionary of unique symbol patterns and their counts.
    """
    # Process the text line by line to exclude book titles and chapter numbers
    lines = text.strip().split('\n')
    processed_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Extract only the main text content, skipping book, chapter, and verse info
        parts = line.split('\t')
        if len(parts) >= 3:  # If the line has tab-separated parts (book, chapter, verse, text)
            # The main text is after the book, chapter, and verse
            main_text = parts[-1]  # Take the last part which should be the text
            processed_lines.append(main_text)
        else:
            # If the line doesn't have the expected format, check if it starts with book code
            match = re.match(r'^[A-Za-z]+\d+[A-Za-z]?-?[A-Za-z]?\(', line)
            if match:
                # This is a book title line, extract only the text part
                main_text = line[match.end()-1:]  # Include the opening parenthesis
                processed_lines.append(main_text)
            else:
                # If we can't identify the format, include the whole line
                processed_lines.append(line)
    
    # Join the processed lines back into a single text
    processed_text = '\n'.join(processed_lines)
    
    # Split the text into words (space-separated)
    words = re.findall(r'\S+', processed_text)
    
    # Collect all symbol patterns between letters
    symbol_patterns = []
    
    for word in words:
        # Find positions of Syriac letters
        letter_positions = [i for i, char in enumerate(word) if is_syriac_letter(char)]
        
        if not letter_positions:
            # If there are no Syriac letters in the word, skip
            continue
            
        # Extract symbol pattern before the first letter
        if letter_positions[0] > 0:
            symbol_pattern = word[:letter_positions[0]]
            symbol_patterns.append(symbol_pattern)
        
        # Extract symbol patterns between consecutive letters
        for i in range(len(letter_positions) - 1):
            start = letter_positions[i] + 1
            end = letter_positions[i + 1]
            
            if start < end:  # If there are symbols between letters
                symbol_pattern = word[start:end]
                symbol_patterns.append(symbol_pattern)
        
        # Extract symbol pattern after the last letter
        if letter_positions[-1] < len(word) - 1:
            symbol_pattern = word[letter_positions[-1] + 1:]
            symbol_patterns.append(symbol_pattern)
    
    # Count unique patterns
    pattern_counts = Counter(symbol_patterns)
    
    return pattern_counts

def main():
    # Read the file
    with open('s3-out-reduced-processed.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Analyze symbol patterns
    pattern_counts = analyze_symbols_between_letters(text)
    
    # Save to CSV file
    with open('patterns.csv', 'w', encoding='utf-8', newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['编号', '符号形式', '频率'])
        
        # Write data rows
        for i, (pattern, count) in enumerate(pattern_counts.most_common(), 1):
            writer.writerow([i, pattern, count])
    
    # Print summary
    print(f"Total unique symbol patterns: {len(pattern_counts)}")
    print(f"Results saved to patterns.csv")

if __name__ == "__main__":
    main()
