"""Module responsible for all kind of caches used by remsen."""
import hashlib
from pathlib import Path
from typing import Optional

import fiona

import geopandas

from remsen import vector


DEFAULT_CACHE_DIR = Path(".cache").resolve()


def sha1_hash(file_path: Path) -> str:
    """Return SHA-1 hash of given file."""
    assert file_path.is_file()
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as file_handler:
        while True:
            data = file_handler.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


class CadastreCache:
    """Cache for given cadastre set."""

    def __init__(
        self,
        cadastre_path: Path,
        name: str,
        layer_name: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Construct cache for given cadastre set."""
        self.sha1_hash = sha1_hash(cadastre_path)
        self.name = name
        parent_dir = cache_dir or DEFAULT_CACHE_DIR
        self.directory = parent_dir / "cadastre" / self.name
        self.layer_name = layer_name
        self.cadastre_path = cadastre_path
        if self.directory.exists():
            import shutil
            shutil.rmtree(str(self.directory))
        if not self.directory.exists():
            self.first_time_setup()

    @classmethod
    def from_name(cls, name: str) -> "CadastreCache":
        pass

    def first_time_setup(self):
        self.directory.mkdir(parents=True)
        with fiona.open(self.cadastre_path, "r", layer=self.layer_name) as f:
            self.dataframe = geopandas.GeoDataFrame(f, crs=4285)

    def __len__(self) -> int:
        """Return number of cadastre in source data."""
        return vector.layer_size(
            path=self.cadastre_path,
            layer=self.layer_name,
        )

    def __repr__(self) -> str:
        """Return pertinent information regarding cadastre cache."""
        return (
            "CadastreCache("
            f'name="{self.name}", '
            f"length={len(self)}, "
            f'path="{self.directory}"'
            ")"
        )
