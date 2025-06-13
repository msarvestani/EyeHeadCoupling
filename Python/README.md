# How to install the head_movement environment
Assuming you have the conda installed, do the following steps in the terminal:
`conda env create -f head_movement_env.yml`

# Running the .ipynb notebook
- Run the `script_after_session.ipynb` file.
- The program will ask you to select a rat/treeshrew session folder
- Then the program will ask you to specify the type of stim used. Get this information from the session powerpoint. Options are: None, LR (Left Right), UD (Up down), Interleaved
- The program will generate plots of the saccade directions of the whole sessions, and 500 ms after the go cue was shown.
