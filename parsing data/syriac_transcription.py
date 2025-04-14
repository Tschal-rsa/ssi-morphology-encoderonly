def remove_diacritics(text):
    """
    Remove all diacritical marks and other non-letter characters from Syriac text.
    """
    # Define the basic Syriac letters
    SYRIAC_LETTERS = {
        'ܐ', 'ܒ', 'ܓ', 'ܕ', 'ܗ', 'ܘ', 'ܙ', 'ܚ', 'ܛ', 'ܝ', 'ܟ', 'ܠ', 'ܡ', 
        'ܢ', 'ܣ', 'ܥ', 'ܦ', 'ܨ', 'ܩ', 'ܪ', 'ܫ', 'ܬ'
    }
    
    # Initialize result
    result = []
    
    # Process each character
    for char in text:
        if char in SYRIAC_LETTERS:
            result.append(char)
        elif char.isspace():
            result.append(char)
    
    return ''.join(result)

def convert_to_syriac(text):
    """
    Convert Syriac text to Latin transcription.
    """
    # First remove all diacritics
    text = remove_diacritics(text)
    
    # Syriac to Latin mapping
    SYRIAC_TO_LATIN = {
        'ܐ': '>',  # ALAF
        'ܒ': 'B',  # BETH
        'ܓ': 'G',  # GAMAL
        'ܕ': 'D',  # DALATH
        'ܗ': 'H',  # HE
        'ܘ': 'W',  # WAW
        'ܙ': 'Z',  # ZAIN
        'ܚ': 'X',  # HETH
        'ܛ': 'V',  # TETH
        'ܝ': 'J',  # YUDH
        'ܟ': 'K',  # KAPH
        'ܠ': 'L',  # LAMADH
        'ܡ': 'M',  # MIM
        'ܢ': 'N',  # NUN
        'ܣ': 'S',  # SEMKATH
        'ܥ': '<',  # E
        'ܦ': 'P',  # PE
        'ܨ': 'Y',  # SADHE
        'ܩ': 'Q',  # QOPH
        'ܪ': 'R',  # RISH
        'ܫ': 'C',  # SHIN
        'ܬ': 'T'   # TAW
    }
    
    # Initialize result
    result = []
    
    # Process each character
    for char in text:
        if char in SYRIAC_TO_LATIN:
            result.append(SYRIAC_TO_LATIN[char])
        else:
            # If character is not in mapping, keep it as is
            result.append(char)
    
    return ''.join(result)

def process_file(input_file, output_file):
    """
    Process the input file and write the transcription to the output file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split the text into lines and process each line
        lines = text.split('\n')
        processed_lines = []
        current_chapter = None
        current_verse = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a chapter line
            if line.startswith('Chapter'):
                # If we have accumulated content, process it before changing chapter
                if current_content and current_chapter and current_verse:
                    content = ' '.join(current_content)
                    transcription = convert_to_syriac(content)
                    processed_lines.append(f"Is\t{current_chapter}\t{current_verse}\t{transcription}")
                    current_content = []
                current_chapter = line.split()[1]
                continue
                
            # Check if this is a verse line
            if line[0].isdigit():
                # If we have accumulated content, process it before changing verse
                if current_content and current_chapter and current_verse:
                    content = ' '.join(current_content)
                    transcription = convert_to_syriac(content)
                    processed_lines.append(f"Is\t{current_chapter}\t{current_verse}\t{transcription}")
                    current_content = []
                
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    current_verse = parts[0]
                    current_content = [parts[1]]
            else:
                # This is a continuation of the current verse
                if current_verse:
                    current_content.append(line)
        
        # Process any remaining content
        if current_content and current_chapter and current_verse:
            content = ' '.join(current_content)
            transcription = convert_to_syriac(content)
            processed_lines.append(f"Is\t{current_chapter}\t{current_verse}\t{transcription}")
        
        # Write the result to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
            
        print(f"Successfully processed {input_file} and wrote transcription to {output_file}")
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")

def main():
    # Process the input file
    input_file = "isaiah.txt"
    output_file = "isaiah_transcription.txt"
    process_file(input_file, output_file)

if __name__ == "__main__":
    main() 