"""Output paths shared by policy training and benchmark workers."""
import os
from pathlib import Path


def training_root(material):
    if material not in ('bcc', 'fcc', 'hcp'):
        raise ValueError('unknown material')
    root=Path(os.environ.get('ATOMWORLD_RESULTS', 'results')).expanduser().resolve()
    return root/material
