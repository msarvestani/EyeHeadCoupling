# Eye-Head Coupling Analysis

This repository analyzes eye and head movement coupling in rats and tree shrews.

## 🧰 Setup Instructions

1. **Clone the repository**

   Open a terminal and run:

   ```
   git clone https://github.com/SarvestaniLab/EyeHeadCoupling.git
   cd EyeHeadCoupling/Python
   ```

2. **Create the conda environment**

   Make sure you have conda installed, then run:

   ```
   conda env create -f environment.yml
   ```

3. **Activate the environment**

   ```
   conda activate EyeHeadCoupling
   ```

## 🚀 Running the Jupyter Notebook

1. Launch the notebook:

   ```
   jupyter lab script_after_session.ipynb
   ```

2. Follow the prompts:
   - Select the session folder (rat or tree shrew).
   - Choose the stimulus type based on the session PowerPoint:
     - None
     - LR (Left–Right)
     - UD (Up–Down)
     - Interleaved

3. The notebook will generate:
   - Saccade direction plots for the full session
   - Saccade direction plots for the 500 ms following the "go" cue

## 🧼 Notes

- If your conda environment is named differently, adjust the activation command accordingly.
- Use feature branches and pull requests to propose changes or additions to the codebase.