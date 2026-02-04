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

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _Array([x + other for x in self])
        return _Array(list(self) + list(other))

    def __gt__(self, other):
        return _Array([x > other for x in self])

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


def _argsort(arr):
    return _Array(sorted(range(len(arr)), key=lambda i: arr[i]))


def _diff(arr):
    return _Array([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])


def _where(cond):
    return (_Array([i for i, x in enumerate(cond) if x]),)

class _R:
    def __getitem__(self, items):
        res = []
        for a in items:
            res.extend(a if isinstance(a, (list, _Array)) else [a])
        return _Array(res)


numpy_stub = types.SimpleNamespace(
    genfromtxt=_genfromtxt,
    argsort=_argsort,
    diff=_diff,
    where=_where,
    r_=_R(),
    ndarray=_Array,
)


class _Config:
    def __init__(self, folder_path: Path) -> None:
        self.folder_path = folder_path
        self.camera_side = None
        self.animal_id = None


def test_cue_arrays_present(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))
    sys.modules["numpy"] = numpy_stub
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *_: None)

    spec = importlib.util.spec_from_file_location("io", root / "eyehead" / "io.py")
    io = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(io)
    load_session_data = io.load_session_data
    cue = tmp_path / "cue.csv"
    cue.write_text("frame,time\n1,0.0\n2,2.0\n5,4.0\n")
    ell = tmp_path / "ellipse_center_xy.csv"
    ell.write_text("frame,time,x,y\n1,0,0,0\n")
    origin = tmp_path / "origin_of_eyecoordinate.csv"
    origin.write_text("frame,time,lx,ly,rx,ry\n1,0,0,0,0,0\n")
    data = load_session_data(_Config(tmp_path))
    assert data.cue_frame is not None
    assert data.cue_time is not None

