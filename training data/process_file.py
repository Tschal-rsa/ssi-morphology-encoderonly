import re

# 定义前缀标记
MC_PREFIXES = ['!', ']', '@']

def mc_reduce(s: str) -> str:
    """
    将输出简化为最小形式。
    简化包括：
    1. 移除所有双重标记前缀中的最左侧标记
    2. 移除冒号后的连续小写字母，但保留冒号本身
    """
    # 对于每个前缀标记，使用正则表达式移除左侧标记
    for c in MC_PREFIXES:
        s = re.sub(f'{c}([^{c}]*{c})', r'\1', s)
    # 移除冒号后的连续小写字母，但保留冒号
    s = re.sub(r':([a-z]+)', ':', s)
    return s

def process_file(input_file: str, output_file: str):
    """
    处理输入文件并将结果写入输出文件
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每一行
        processed_lines = [mc_reduce(line.strip()) for line in lines]
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
                
        print(f"处理完成！结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

if __name__ == "__main__":
    input_file = "s3-out-reduced=:(.txt"
    output_file = "s3-out-reduced-processed.txt"
    process_file(input_file, output_file) 