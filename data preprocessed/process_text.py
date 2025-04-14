import csv
import re

def load_patterns(patterns_file):
    patterns = {}
    with open(patterns_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patterns[row['符号形式']] = row['编号']
    return patterns

def is_syriac_or_space(char):
    # 检查是否为叙利亚文字母（A-Z或<>）或空格
    return bool(re.match(r'[A-Z<> ]', char))

def process_text(input_file, patterns_file, output_file):
    # 加载模式
    patterns = load_patterns(patterns_file)
    
    # 将模式按长度降序排序，这样较长的模式会优先匹配
    sorted_patterns = sorted(patterns.keys(), key=len, reverse=True)
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行，提取正文部分
    processed_lines = []
    for line in lines:
        # 分割行，只取第四列（正文部分）
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            text = parts[3]
            # 将文本拆分成单个字符
            chars = list(text)
            processed_lines.extend(chars)
            # 添加两个空格表示换行
            processed_lines.extend([' ', ' '])
    
    # 首先获取原始的标记结果
    result = []
    i = 0
    while i < len(processed_lines):
        # 检查当前字符是否可能是模式的开始
        found_pattern = False
        # 使用排序后的模式列表进行匹配
        for pattern in sorted_patterns:
            if i + len(pattern) <= len(processed_lines):
                current_text = ''.join(processed_lines[i:i+len(pattern)])
                if current_text == pattern:
                    # 将模式中的所有字符都添加到结果中，并标记相应的编号
                    for char in pattern:
                        result.append([char, patterns[pattern]])
                    i += len(pattern)
                    found_pattern = True
                    break
        
        if not found_pattern:
            # 如果没有找到模式，添加当前字符，output设为0
            result.append([processed_lines[i], '0'])
            i += 1
    
    # 处理标记结果：合并数字到前面的字母/空格
    final_result = []
    pending_numbers = set()  # 用于存储待合并的数字
    last_valid_char = None
    
    for char, number in result:
        if is_syriac_or_space(char):
            # 如果有待合并的数字且当前是新的字母/空格
            if pending_numbers and last_valid_char is not None:
                # 将之前累积的数字合并到上一个字母/空格
                if len(pending_numbers) > 1:
                    print(f"错误：字符 '{last_valid_char}' 需要接收多个不同的数字: {pending_numbers}")
                    return
                final_result[-1][1] = ','.join(sorted(pending_numbers))
                pending_numbers.clear()
            
            final_result.append([char, '0'])
            last_valid_char = char
        elif number != '0':
            pending_numbers.add(number)
    
    # 处理最后一组待合并的数字
    if pending_numbers and last_valid_char is not None:
        if len(pending_numbers) > 1:
            print(f"错误：字符 '{last_valid_char}' 需要接收多个不同的数字: {pending_numbers}")
            return
        final_result[-1][1] = ','.join(sorted(pending_numbers))
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['input', 'output'])
        # 写入数据
        writer.writerows(final_result)

if __name__ == "__main__":
    process_text('s3-out-reduced-processed.txt', 'patterns.csv', 'output.csv') 