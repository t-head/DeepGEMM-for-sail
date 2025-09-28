from utils import parse_dump_file
import os

def remove_duplicate_lines(input_file, output_file):
    """
    Reads lines from the input file, removes duplicates, and writes them to the output file.
    """
    try:
        # 使用集合来自动去重行
        unique_lines = list()
        line_idx_pair = {}

        # 读取输入文件
        with open(input_file, 'r') as infile:
            idx = 1
            for line in infile:
                # 去掉每行末尾的回车并添加到output
                line = line.strip()
                if line == "" or line.startswith("#"):
                    unique_lines.append(line)
                elif line not in unique_lines:
                    unique_lines.append(line)
                    line_idx_pair[line] = idx
                else:
                    print(f"duplicate line {idx} with {line_idx_pair[line]}")
                idx += 1

        # 将去重后的行写入输出文件
        with open(output_file, 'w') as outfile:
            for line in unique_lines:
                outfile.write(line + '\n')

        print(f"去重后的内容已写入到 {output_file}")

    except FileNotFoundError:
        print(f"文件 {input_file} 未找到。请检查输入路径。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--input', default=None, type=str, required=True, help='the list of DG cases')
    parser.add_argument('--output', default=None, type=str, required=True, help='output filename')

    args = parser.parse_args()

    remove_duplicate_lines(args.input, args.output)