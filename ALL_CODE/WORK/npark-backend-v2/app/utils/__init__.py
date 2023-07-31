import os
import json

from pathlib import Path


def get_file_size(path):
    return os.path.getsize(path)

def get_folder_size(folder):
    root_directory = Path(folder)
    total_size = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    return total_size

def mkdir(path=None, paths=None):
    if not (path or paths):
        return False
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
    if paths:
        for el in paths:
            if not os.path.exists(el):
                os.makedirs(el)
    return True

def geom_to_file(geom, path):
    with open(path, 'w') as file:
        file.write(json.dumps(geom))
    return True
