"""Download a released checkpoint bundle and verify its SHA256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        p = (root / row['path']).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes checkpoint directory')
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                h.update(b)
        if p.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {p}')
    print(f"Verified {len(manifest['files'])} files")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', default='yingheng/LAP-checkpoints', help='Hugging Face model repository')
    p.add_argument('--revision', default='main', help='Use an immutable commit for reproducibility')
    p.add_argument('--output', type=Path, default=Path('checkpoints'))
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if not a.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(a.repo_id, revision=a.revision, local_dir=a.output,
                          allow_patterns=['manifest.json','diffusion/*.pt','assets/*.pt','README.md'])
    verify(a.output)


if __name__ == '__main__':
    main()
