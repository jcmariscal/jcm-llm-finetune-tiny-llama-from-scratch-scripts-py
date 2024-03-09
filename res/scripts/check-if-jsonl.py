#!/usr/bin/env python3
import json
import sys
import os

def is_jsonl_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if not stripped_line:
                    return False
                json.loads(stripped_line)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False

def check_path(path):
    if os.path.isfile(path):
        if is_jsonl_file(path):
            print(f"The file {path} is in JSONL format.")
        else:
            print(f"The file {path} is not in JSONL format.")
    elif os.path.isdir(path):
        # Sort the list of files before processing
        files = sorted(os.listdir(path))
        for filename in files:
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                if is_jsonl_file(file_path):
                    print(f"The file {file_path} is in JSONL format.")
                else:
                    print(f"The file {file_path} is not in JSONL format.")
    else:
        print(f"The path {path} is neither a file nor a directory.")

def main():
    if len(sys.argv) != 2:
        print("Usage: {} <file_path_or_directory_path>".format(sys.argv[0]))
        sys.exit(1)

    path = sys.argv[1]
    check_path(path)

if __name__ == "__main__":
    main()
