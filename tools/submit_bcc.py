"""Submit Full BCC policy training on a Slurm cluster."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--tag', default='atomworld')
    parser.add_argument('--partition')
    parser.add_argument('--episode', type=positive_int, required=True)
    parser.add_argument('--rounds', type=positive_int, default=25)
    parser.add_argument('--cu', type=positive_int, required=True)
    parser.add_argument('--size', type=positive_int, default=50)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('tag must be a directory name')
    if args.cu + 4 > 2 * args.size ** 3:
        parser.error('Cu atoms and four vacancies exceed the lattice size')
    root = Path(args.root).resolve(); project = Path(args.project).resolve()
    root.mkdir(parents=True, exist_ok=True); (root/'logs').mkdir(exist_ok=True)
    manifest = root/'jobs.json'
    if manifest.exists():
        raise FileExistsError('run already submitted')
    env = os.environ.copy()
    env.update(ATOMWORLD_RUN_ROOT=str(root), ATOMWORLD_PROJECT=str(project),
               ATOMWORLD_RESULTS=str(root/'models'))
    command = ['sbatch', '--parsable', '--export=ALL']
    if args.partition:
        command += ['--partition', args.partition]
    command += [str(project/'slurm/bcc.sbatch'), '--episode', str(args.episode),
                '--rounds', str(args.rounds), '--cu', str(args.cu), '--size', str(args.size),
                '--seed', str(args.seed), '--tag', args.tag]
    job = subprocess.check_output(command, cwd=root, env=env, text=True).strip().split(';')[0]
    sources = {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('atomworld', 'slurm') for p in (project/folder).rglob('*')
               if p.is_file() and p.suffix in ('.py', '.sbatch') and '__pycache__' not in p.parts}
    manifest.write_text(json.dumps({'job_id':job, 'parameters':vars(args),
                                    'source_sha256':sources}, indent=2), encoding='utf-8')
    print(job)


if __name__ == '__main__': main()
