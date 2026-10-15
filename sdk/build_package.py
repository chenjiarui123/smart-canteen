"""Build wheel and source archive using already installed setuptools/wheel."""
import hashlib
import json
import os
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel

root = Path(__file__).resolve().parent
os.chdir(root)
package = root / "src" / "smart_canteen_sdk"
manifest = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((package / "_vendor").rglob("*")) if p.suffix in {".py", ".json"}}
(package / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
dist = root / "dist"
dist.mkdir(exist_ok=True)
wheel = build_wheel(str(dist))
source = build_sdist(str(dist))
checksums = {name: hashlib.sha256((dist / name).read_bytes()).hexdigest() for name in [wheel, source]}
(dist / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(json.dumps(checksums, indent=2))
