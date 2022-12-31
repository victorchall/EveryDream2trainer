import os
import argparse
import hashlib

def hash_bins(path):
    hash = hashlib.sha256()
    #recursively walk through the path
    for root, dirs, files in os.walk(path):
        for file in files:
            #check if file is a .py file
            if file.endswith('.bin'):
                #sha256 hash the file
                with open(os.path.join(root, file), 'rb') as f:
                    hash.update(f.read())
                    #write the hash to a file
                    
                    #get filename without extension
                    filename = os.path.splitext(file)[0]
                    
                    with open(f'{filename}.sha256', 'w') as hf:
                        hf.write(hash.hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True, help="Path ")
    hash_bins(path)