
import sys
from datasets import load_dataset
print('Starting load...', file=sys.stderr)
try:
    ds = load_dataset('bh2821/LightNovel5000', split='train', trust_remote_code=True)
    print(f'Columns: {ds.column_names}', file=sys.stderr)
    if len(ds) > 0:
        print(f'First example keys: {ds[0].keys()}', file=sys.stderr)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
