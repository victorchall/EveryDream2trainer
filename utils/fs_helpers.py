import logging 

def barename(file):
    (val, _) = os.path.splitext(os.path.basename(file)) 
    return val

def ext(file):
    (_, val) = os.path.splitext(os.path.basename(file)) 
    return val.lower()

def same_barename(lhs, rhs):
    return barename(lhs) == barename(rhs)
    
def is_image(file):
    return ext(file) in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif'}

def read_text(file):
    try:
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin-1']
        for encoding in encodings:
            try:
                with open(file, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f'Could not decode file with any of the provided encodings: {encodings}')
    except Exception as e:
        logging.warning(f" *** Error reading text file: {file}: {e}")

def read_float(file):
    try:
        return float(read_text(file))
    except Exception as e:
        logging.warning(f" *** Could not parse to float in file {file}: {e}")

import os

def walk_and_visit(path, visit_fn, context=None):
    names = [entry.name for entry in os.scandir(path)]

    dirs = []
    files = []
    for name in names:
        fullname = os.path.join(path, name)

        if str(name).startswith('.'):
            continue

        if os.path.isdir(fullname):
            dirs.append(fullname)
        else:
            files.append(fullname)

    subcontext = visit_fn(files, context)

    for subdir in dirs:
        walk_and_visit(subdir, visit_fn, subcontext)