"""Build a source archive with a file checksum manifest."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    excluded={'__pycache__','.pytest_cache','.git','.venv','build','dist','artifacts','results','runs','logs'}
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and
           not any(x in excluded or x.endswith('.egg-info') for x in p.relative_to(root).parts) and
           p.suffix not in ('.pyc','.pt','.npz','.zip','.gz') and p.name!='MANIFEST.sha256']
    manifest=root/'MANIFEST.sha256'
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files),encoding='utf-8')
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[manifest]:archive.write(path,'AtomWorld/'+path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for path in files:
            assert hashlib.sha256(archive.read('AtomWorld/'+path.relative_to(root).as_posix())).digest()==hashlib.sha256(path.read_bytes()).digest()
    print(output)


if __name__=='__main__':main()
