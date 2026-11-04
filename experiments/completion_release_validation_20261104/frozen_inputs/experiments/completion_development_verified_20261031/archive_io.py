"""Local long-path I/O for complete, unchanged experiment input archives."""
import hashlib
import os
from pathlib import Path
import shutil


def disk_path(path):
    path = Path(path).absolute()
    text = str(path)
    if os.name == 'nt' and not text.startswith('\\\\?\\'):
        if text.startswith('\\\\'):
            text = '\\\\?\\UNC\\'+text[2:]
        else:
            text = '\\\\?\\'+text
        return Path(text)
    return path


def sha(path):
    return hashlib.sha256(disk_path(path).read_bytes()).hexdigest()


def verify_files(files, root):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('Frozen file drift: '+name)


def copy_original(source, destination, expected):
    source, destination = disk_path(source), disk_path(destination)
    if sha(source) != expected:
        raise RuntimeError('Archive source drift: '+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != expected:
            raise RuntimeError('Existing archive drift: '+str(destination))
    else:
        shutil.copyfile(source, destination)
    if sha(destination) != expected:
        raise RuntimeError('Archive copy drift: '+str(destination))
