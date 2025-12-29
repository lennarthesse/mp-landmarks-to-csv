#import landmark_extractor
import os

def load_files_from_directory(path):
    return os.scandir(path)

if __name__ == "__main__":
    files = load_files_from_directory(".")
    for file in files:
        print(file.name)
