
from datasets import load_dataset
import sys

def check_dataset(name):
    print(f"Checking {name}...", file=sys.stderr)
    try:
        ds = load_dataset(name, split="train", trust_remote_code=True)
        print(f"Columns for {name}: {ds.column_names}", file=sys.stderr)
        if len(ds) > 0:
            print(f"First example keys for {name}: {ds[0].keys()}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading {name}: {e}", file=sys.stderr)

check_dataset("pleisto/wikipedia-cn-20230720-filtered")
