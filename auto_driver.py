import os
import subprocess
import time

'''
Script to step through configs and train them NOT all at the same time, so I can leave this running and go to bed without crashing my laptop
'''

# this copy of script runs every 3rd or every 4th or whatever. Edit as needed. (so can run this script 2x or 3x in parallel)
mod_fact = 0
mod_base = 2 # 2 to do even/odd, or 3 to do every 3rd, 5 for every 5th etc

# Path to the directory to check
dir_path = '/Users/nick/dq_runs/'

# Path to the script to run
script_path = 'deepQsolver.py'

dirs = os.listdir(dir_path)
dirs.sort(reverse=True)
# Iterate over all subdirectories in the directory
for subdir in dirs:
    subdir_path = os.path.join(dir_path, subdir)
    if subdir.startswith('config_'):

        # Check if the subdirectory contains a file named 'model.pth'
        if 'model.pth' not in os.listdir(subdir_path):
            id_number = int(subdir.split('_')[-1])
            if id_number % mod_base == mod_fact:
                print(f"Missing model.pth in {subdir_path}, id number {id_number}")
                print(f"Running {subdir_path}")
                this_cmd = script_path + '' + str(id_number)
                subprocess.call(['python3', script_path, str(id_number)])
                # Wait for the script to finish before moving on to the next subdirectory

