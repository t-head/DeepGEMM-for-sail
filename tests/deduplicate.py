from utils import parse_dump_file
import os

def remove_duplicate_lines(input_file, output_file):
    """
    Reads lines from the input file, removes duplicates, and writes them to the output file.
    """
    try:
        unique_lines = list()
        line_idx_pair = {}

        with open(input_file, 'r') as infile:
            idx = 1
            for line in infile:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    unique_lines.append(line)
                elif line not in unique_lines:
                    unique_lines.append(line)
                    line_idx_pair[line] = idx
                else:
                    print(f"duplicate line {idx} with {line_idx_pair[line]}")
                idx += 1

        with open(output_file, 'w') as outfile:
            for line in unique_lines:
                outfile.write(line + '\n')

        print(f"Deduplicated content written to {output_file}")

    except FileNotFoundError:
        print(f"File {input_file} not found. Please check the input path.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--input', default=None, type=str, required=True, help='the list of DG cases')
    parser.add_argument('--output', default=None, type=str, required=True, help='output filename')

    args = parser.parse_args()

    remove_duplicate_lines(args.input, args.output)