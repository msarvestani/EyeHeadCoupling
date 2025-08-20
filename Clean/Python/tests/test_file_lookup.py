from pathlib import Path
import importlib.util
import sys
import types


class _Array(list):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            rows = range(len(self)) if rows == slice(None) else rows
            if isinstance(rows, slice):
                rows = range(*rows.indices(len(self)))
            elif isinstance(rows, int):
                rows = [rows]
            if isinstance(cols, slice):
                return _Array([
                    _Array(self[r][c] for c in range(*cols.indices(len(self[r]))))
                    for r in rows
                ])
            return _Array([self[r][cols] for r in rows])
        if isinstance(key, slice):
            return _Array(super().__getitem__(key))
        if isinstance(key, list):
            return _Array([self[i] for i in key])
        return super().__getitem__(key)

    def astype(self, dtype):
        return _Array([dtype(x) for x in self])

    @property
    def shape(self):
        if self and isinstance(self[0], (list, _Array)):
            return (len(self), len(self[0]))
        return (len(self),)


def _genfromtxt(path, delimiter=",", skip_header=1):
    if hasattr(path, "read"):
        content = path.read().splitlines()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.readlines()
    lines = [line.strip() for line in content[skip_header:] if line.strip()]
    return _Array([_Array(float(x) for x in line.split(delimiter)) for line in lines])


numpy_stub = types.SimpleNamespace(genfromtxt=_genfromtxt, ndarray=_Array)


class _Config:
    def __init__(self, folder_path: Path, animal_id: str | None = None) -> None:
        self.folder_path = folder_path
        self.camera_side = None
        self.animal_id = animal_id


def _load_session(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))
    sys.modules["numpy"] = numpy_stub
    spec = importlib.util.spec_from_file_location("io", root / "eyehead" / "io.py")
    io = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(io)
    return io.load_session_data


def _write_required(tmp_path: Path) -> None:
    (tmp_path / "ellipse_center_xy.csv").write_text("frame,time,x,y\n1,0,0,0\n")
    (tmp_path / "origin_of_eyecoordinate.csv").write_text(
        "frame,time,lx,ly,rx,ry\n1,0,0,0,0,0\n"
    )


def test_prefers_prefixed_file(tmp_path: Path) -> None:
    _write_required(tmp_path)
    (tmp_path / "animal_camera.csv").write_text("frame,time\n1,0\n")
    (tmp_path / "camera.csv").write_text("frame,time\n2,0\n")
    load_session_data = _load_session(tmp_path)
    data = load_session_data(_Config(tmp_path, "animal"))
    assert data.camera is not None
    assert data.camera[0][0] == 1.0


def test_falls_back_to_unprefixed(tmp_path: Path) -> None:
    _write_required(tmp_path)
    (tmp_path / "camera.csv").write_text("frame,time\n5,0\n")
    load_session_data = _load_session(tmp_path)
    data = load_session_data(_Config(tmp_path, "animal"))
    assert data.camera is not None
    assert data.camera[0][0] == 5.0
