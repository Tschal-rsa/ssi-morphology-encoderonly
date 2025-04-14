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
        
        # Replace all newlines with spaces
        text = text.replace('\n', ' ')
        
        # Convert the text
        transcription = convert_to_syriac(text)
        
        # Write the result to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
            
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