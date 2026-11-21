"""Copy the validated static application into its hosting output directory."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dist'
OUT.mkdir(exist_ok=True)
for pattern in ('*.html', '*.css', '*.js'):
    for source in ROOT.glob(pattern):
        shutil.copyfile(source, OUT / source.name)
shutil.copytree(ROOT / 'assets', OUT / 'assets', dirs_exist_ok=True)
print(f'Packaged {OUT}')
