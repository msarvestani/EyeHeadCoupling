# Eye-Head Coupling Analysis

Tools for studying the relationship between eye and head movements in rats and tree shrews.

## Directory Structure
- `MATLAB/` – legacy MATLAB code.
- `Python/` – all Python analysis code, utilities and notebooks.
- `session_manifest.yml` – session configuration file in the root directory.

## Setup
1. Clone the repository
   ```
   git clone https://github.com/SarvestaniLab/EyeHeadCoupling.git
   cd EyeHeadCoupling
   ```
2. Create the conda environment
   ```
   conda env create -f Python/EyeHeadCoupling.yml
   ```
3. Activate the environment
   ```
   conda activate EyeHeadCoupling
   ```

## Session Manifest
Session metadata lives in `session_manifest.yml` and maps session identifiers to their settings:
```yaml
sessions:
  session_01:
    session_path: /path/to/session_01
    results_dir: /path/to/session_01/results
```

Use `utils.session_loader.load_session` to access entries.

## Usage
- Run analysis scripts from `Python/analysis/`.
- Launch Jupyter notebooks from `Python/notebooks/`.

## Plotting style
Matplotlib figures use a repository-wide style defined in `Python/style.mplstyle`.
The helper functions in `Python/eyehead/plotting.py` load this file with
``plt.style.use`` so that all plots share consistent fonts and colours.
