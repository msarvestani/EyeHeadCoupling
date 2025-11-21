# CLAUDE.md - AI Assistant Guide for EyeHeadCoupling Repository

## Repository Overview

**Purpose**: Analysis framework for studying coordinated eye and head movements in tree shrews and rats during behavioral tasks.

**Primary Language**: Python 3.10

**Research Context**: This repository analyzes Bonsai-generated experimental data from fixation, prosaccade, and antisaccade tasks to understand oculomotor control and eye-head coordination.

**Animals**: Tree shrews (Paris/Tsh001, Apollo/Tsh002)

---

## Directory Structure

```
EyeHeadCoupling/
├── Clean/                          # Primary working directory (USE THIS)
│   ├── data/
│   │   └── session_manifest.yml    # Session metadata and configuration
│   ├── MATLAB/                     # Legacy MATLAB code (reference only)
│   └── Python/                     # Main Python codebase
│       ├── eyehead/                # Core package
│       │   ├── analysis.py         # Saccade detection, calibration, analysis
│       │   ├── io.py               # Data loading and file I/O
│       │   ├── plotting.py         # Visualization utilities
│       │   ├── filters.py          # Signal processing
│       │   └── ui.py               # GUI dialogs
│       ├── analysis/               # Analysis scripts
│       │   ├── fixation_session.py
│       │   ├── fixation_population.py
│       │   ├── prosaccade_session.py
│       │   └── prosaccade_population.py
│       ├── utils/
│       │   └── session_loader.py   # Session config management
│       ├── tests/                  # Test suite
│       └── notebooks/              # Jupyter notebooks
├── Python/                         # Legacy/alternate location (avoid)
└── README.md
```

**Important**: All development should focus on `Clean/Python/`. The root `Python/` directory appears to be legacy code.

---

## Development Setup

### Environment Setup

```bash
# Create conda environment from specification
conda env create -f Clean/Python/EyeHeadCoupling.yml

# Activate environment
conda activate EyeHeadCoupling
```

### Key Dependencies

- **Computer Vision**: opencv
- **Scientific Computing**: numpy, scipy, pandas, scikit-learn
- **Visualization**: matplotlib (with custom style), ipympl
- **Notebook Environment**: jupyterlab, ipykernel
- **GUI**: pyqt

### Custom Plotting Style

The repository uses a custom matplotlib style defined in `Clean/Python/style.mplstyle`:
- Font: Arial, size 12
- Colormap: viridis
- Automatically loaded by importing `eyehead.plotting`

---

## Core Package: `eyehead`

### `eyehead.analysis` (1,341 lines)

**Key Functions**:

- `calibrate_eye_position(eye_pos, calibration_factor)` - Applies calibration to raw eye coordinates
- `detect_saccades(timestamps, eye_pos, config)` - Detects saccades using velocity thresholds
- `organize_stims(stim_data, timestamps, max_interval_s)` - Organizes stimulus presentations into trials
- `sort_saccades(saccades, stim_data)` - Classifies saccades relative to stimuli
- `plot_fixation_intervals_by_trial()` - Visualizes fixation periods
- `quantify_fixation_stability_vs_random()` - Statistical comparison

**Configuration**:
```python
@dataclass
class SaccadeConfig:
    saccade_threshold: float = 3.0      # deg/s
    saccade_threshold_torsion: float = 1.5  # deg/s
    blink_threshold: float = 10.0       # deg/s
    blink_detection: int = 1            # 0=off, 1=on
    saccade_win: float = 0.7            # seconds
```

### `eyehead.io` (258 lines)

**Key Classes**:
```python
@dataclass
class SessionData:
    camera: pd.DataFrame
    imu: pd.DataFrame  # Head movement data
    cue: Optional[pd.DataFrame]
    go: Optional[pd.DataFrame]
    end_of_trial: Optional[pd.DataFrame]
    ellipse_center_xy_L: Optional[pd.DataFrame]
    ellipse_center_xy_R: Optional[pd.DataFrame]
    # ... other eye tracking fields
```

**Key Functions**:
- `load_session_data(session_path, animal_id, camera_side)` - Loads all CSV files for a session
- File discovery uses fallback patterns: `{animal_id}_{filename}.csv` → `{filename}.csv`

### `eyehead.plotting` (87 lines)

**Key Functions**:
- `plot_polar_angle_distribution()` - Circular histograms for directional data
- `angle_to_color()` - Maps angles to colors for visualization
- Style automatically applied on import

### `utils.session_loader` (276 lines)

**Key Classes**:
```python
@dataclass
class SessionConfig:
    session_name: str
    experiment_type: str  # 'fixation', 'prosaccade', 'antisaccade', 'fixation-control'
    session_path: Path
    date: datetime
    animal_name: str
    animal_id: str
    calibration_factor: float
    ttl_freq: int
    camera_side: str  # 'L' or 'R'
    saccade_config: SaccadeConfig
    # ... other fields
```

**Key Functions**:
- `load_session_manifest(manifest_path)` - Parses YAML manifest
- `filter_sessions(sessions, experiment_type, animal_name)` - Filters by criteria

---

## Data File Formats

### CSV Files (Bonsai-generated)

**Preprocessing**: Files undergo automatic cleaning:
- Remove parentheses: `(value)` → `value`
- Convert booleans: `True` → `1`, `False` → `0`

**File Types**:
- `camera.csv` - Frame timing and metadata
- `go.csv` - "Go" cue timing and direction
- `ellipse_center_xy_{L|R}.csv` - Eye position (x, y coordinates)
- `origin_of_eyecoordinate_{L|R}.csv` - Eye coordinate system origin
- `torsion_{L|R}.csv` - Torsional eye movement
- `vdaxis_{L|R}.csv` - Vertical-dorsal axis
- `imu.csv` - Head movement (inertial measurement unit)
- `end_of_trial.csv` - Trial outcomes
- `cue.csv` - Visual cue timing and direction

**Naming Convention**:
- Preferred: `{animal_id}_{filename}_{side}.csv`
- Fallback: `{filename}_{side}.csv`
- Side: `L` (left) or `R` (right) for eye-specific data

### Session Manifest (YAML)

**Location**: `Clean/data/session_manifest.yml`

**Structure**:
```yaml
results_root: X:\Analysis\EyeHeadCoupling
max_interval_s: 1.0  # Maximum interval for trial organization

saccade_config:
  saccade_threshold: 3.0
  saccade_threshold_torsion: 1.5
  blink_threshold: 10.0
  blink_detection: 1
  saccade_win: 0.7

sessions:
  session_XX:
    experiment_type: fixation  # or prosaccade, antisaccade, fixation-control
    session_path: X:\Experimental_Data\...
    date: 2025-07-15
    animal_name: Paris
    animal_id: Tsh001
    calibration_factor: 3.76
    ttl_freq: 60
    camera_side: L
    reward_contingency:
      reward_angle: 35
      reward_window: 0.7
    params:
      saccade_config:
        saccade_threshold: 1.0  # Session-specific override
```

**Sessions**: 36 sessions defined (session_08 through session_36)

---

## Analysis Workflows

### Single-Session Analysis

#### Fixation Task
```python
from analysis.fixation_session import main as analyze_fixation

# Analyze single session
results_df = analyze_fixation(
    session_config,
    save_figures=True,
    output_dir="./results"
)
```

**Output**: DataFrame with fixation metrics per session

#### Prosaccade Task
```python
from analysis.prosaccade_session import main as analyze_prosaccade

analyze_prosaccade(
    session_config,
    save_figures=True,
    output_dir="./results"
)
```

**Output**: Directional saccade plots

### Population Analysis

#### Fixation (Multiple Sessions)
```python
from analysis.fixation_population import main as analyze_fixation_pop

results_df = analyze_fixation_pop(
    manifest_path="Clean/data/session_manifest.yml",
    experiment_type="fixation",
    animal_name="Paris"  # or None for all animals
)
```

#### Prosaccade (Multiple Sessions)
```python
from analysis.prosaccade_population import main as analyze_prosaccade_pop

analyze_prosaccade_pop(
    manifest_path="Clean/data/session_manifest.yml",
    experiment_type="prosaccade",
    animal_name="Apollo"
)
```

---

## Testing

### Running Tests

```bash
# Run all tests
cd Clean/Python
pytest tests/

# Run specific test file
pytest tests/test_session_loader.py
```

### Test Coverage

- **test_session_loader.py** - Session configuration loading
- **test_file_lookup.py** - CSV file discovery
- **test_cue_loading.py** - Cue data loading
- **test_end_of_trial_loading.py** - Trial outcome data

---

## Key Conventions

### Code Style

1. **Dataclasses**: Use `@dataclass` for configuration and data structures
2. **Type Hints**: Include type annotations for function parameters and returns
3. **Docstrings**: Document complex functions with purpose, parameters, and returns
4. **Imports**: Group by standard library, third-party, local modules

### Analysis Patterns

1. **Load session config** from YAML manifest
2. **Load session data** using `eyehead.io.load_session_data()`
3. **Calibrate eye position** with `calibrate_eye_position()`
4. **Detect saccades** with `detect_saccades()`
5. **Organize stimuli** with `organize_stims()`
6. **Generate visualizations** using functions in `eyehead.plotting`
7. **Return results** as pandas DataFrames when appropriate

### Data Processing

1. **Non-causal filtering**: Use `eyehead.filters.butter_filter_non_causal()` for smooth trajectories
2. **NaN handling**: Use `eyehead.filters.interpolate_nans()` before filtering
3. **Velocity calculation**: Differentiate position to get velocity for saccade detection
4. **Trial alignment**: Align events to cue and go signals

---

## Git Workflow

### Branch Naming Convention

- Feature branches: `claude/claude-md-{session_id}-{unique_id}`
- CRITICAL: Branch names must start with `claude/` and end with matching session ID

### Commit Guidelines

1. **Clear messages**: Describe what changed and why
2. **Logical grouping**: Group related changes in single commits
3. **Test before commit**: Ensure tests pass

### Push Protocol

```bash
# Always use -u flag for new branches
git push -u origin claude/claude-md-{session_id}-{unique_id}

# Retry on network failures with exponential backoff
# 2s, 4s, 8s, 16s intervals
```

---

## Common Tasks for AI Assistants

### Adding New Analysis Functions

1. **Location**: Add to `Clean/Python/eyehead/analysis.py` or create new module
2. **Pattern**: Follow existing function signatures with `SessionData` or `SessionConfig` inputs
3. **Testing**: Add corresponding test in `tests/`
4. **Documentation**: Update this file with new function description

### Modifying Session Manifest

1. **File**: `Clean/data/session_manifest.yml`
2. **Validation**: Ensure all required fields are present
3. **Testing**: Run `tests/test_session_loader.py` after changes

### Adding New Experiment Type

1. Add to `experiment_type` enum in session config
2. Create analysis script in `Clean/Python/analysis/`
3. Update manifest with sessions of new type
4. Add tests for new experiment type

### Debugging Data Loading Issues

1. **Check file naming**: Verify CSV files follow `{animal_id}_{filename}.csv` pattern
2. **Check manifest**: Ensure `animal_id` and `session_path` are correct
3. **Run tests**: Use `test_file_lookup.py` to debug file discovery
4. **Inspect raw CSV**: Verify format matches expected structure

### Creating Visualizations

1. **Import style**: Always import `eyehead.plotting` to apply custom style
2. **Save figures**: Use `plt.savefig()` with DPI=300 for publication quality
3. **Color conventions**: Use `angle_to_color()` for directional data
4. **Figure size**: Default is set in `style.mplstyle`, override if needed

---

## Important Notes

### Do Not Modify

- **MATLAB code** in `Clean/MATLAB/` - Legacy reference only
- **Environment file** without coordination - Changes affect all users
- **Session dates** in manifest - These are historical records

### Data Paths

- Manifest references Windows paths (`X:\`) - These are for reference
- Local analysis uses paths relative to session config
- Update `results_root` in manifest if changing output location

### Performance Considerations

- **Large sessions**: Some CSV files exceed 100MB
- **Population analysis**: Processing 36 sessions can take minutes
- **Memory**: Load sessions one at a time for population analysis

### Known Patterns

1. **Session naming**: `session_08` through `session_36` (some gaps/commented out)
2. **Camera side**: Most sessions use left eye (`L`), some use right (`R`)
3. **Calibration factors**: Vary by session (typically ~3.76)
4. **Date range**: July 2025 - October 2025

---

## Quick Reference

### File Locations
- Main package: `Clean/Python/eyehead/`
- Analysis scripts: `Clean/Python/analysis/`
- Session manifest: `Clean/data/session_manifest.yml`
- Tests: `Clean/Python/tests/`
- Custom style: `Clean/Python/style.mplstyle`

### Key Entry Points
- Single fixation: `analysis.fixation_session.main()`
- Population fixation: `analysis.fixation_population.main()`
- Single prosaccade: `analysis.prosaccade_session.main()`
- Population prosaccade: `analysis.prosaccade_population.main()`

### Configuration
- Global config: `session_manifest.yml` top-level
- Session-specific: Under each `sessions.session_XX`
- Saccade detection: `saccade_config` parameters

---

## Version Control Status

- **Repository**: Active Git repository
- **Current Branch**: `claude/claude-md-mi97y83ysm4rlle0-01E4v3EAzCNHSCjrXsSW5Tof`
- **Recent Activity**: Adding session date labels to figures, merging PRs
- **Main Branch**: (configured in remote)

---

## Additional Resources

- **Primary Documentation**: `Clean/README.md` - Most comprehensive
- **Python-specific**: `Python/README.md` - Basic setup and usage
- **Root README**: `README.md` - Minimal description

---

**Last Updated**: 2025-11-21
**Repository Version**: Based on commit b1b3570
