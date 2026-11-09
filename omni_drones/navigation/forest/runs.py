"""训练/评测共用的进度、指纹与原子写入；不依赖仿真。"""
import hashlib
import json
import time
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dataset_signature(dataset, rows):
    """检查 XYZ 数据契约，并把图片、观测和真值内容纳入恢复指纹。"""
    from .geometry import PROTOCOL_VERSION
    from .observation import representation_spec
    dataset = Path(dataset)
    manifest = json.loads((dataset/'manifest.json').read_text())
    if (manifest.get('protocol_version') != PROTOCOL_VERSION or
            manifest.get('representation') != representation_spec()):
        raise ValueError('dataset uses an incompatible observation/path protocol; rebuild from raw scans')
    if not manifest.get('complete'):
        raise ValueError('dataset conversion is incomplete')
    paths = set()
    for row in rows:
        if row.get('protocol_version') != PROTOCOL_VERSION:
            raise ValueError(f'incompatible sample protocol: {row.get("id")}')
        paths.update(row[key] for key in ('image', 'observation', 'truth'))
    for name in {row['truth'] for row in rows}:
        truth = json.loads((dataset/name).read_text())
        if 'mesh_file' in truth:
            paths.add(truth['mesh_file'])
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode())
        digest.update(file_hash(dataset/name).encode())
    return dict(manifest=file_hash(dataset/'manifest.json'), assets=digest.hexdigest())


def model_signature(path):
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return dict(path=str(root), files={p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.iterdir()) if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin', '.jinja'}})


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class Progress:
    def __init__(self, label, total, initial=0, every=20):
        self.label, self.total, self.initial, self.every = label, total, initial, every
        self.started = time.perf_counter()
        self.update(initial)

    def update(self, done, detail=''):
        if done not in (self.initial, self.total) and done % self.every:
            return
        elapsed = time.perf_counter()-self.started
        eta = elapsed*(self.total-done)/(done-self.initial) if done > self.initial else 0
        print(f'{self.label} {done}/{self.total} elapsed={elapsed:.0f}s eta={eta:.0f}s {detail}', flush=True)


def read_predictions(path):
    """只修复末尾未提交的一行；中间损坏的数据必须报错。"""
    if not path.exists():
        return []
    rows = []
    with path.open('rb+') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                f.truncate(start)
                break
            rows.append(json.loads(line))
    return rows
