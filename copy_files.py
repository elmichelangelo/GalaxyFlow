import os
import shutil
import glob

# Specify the directories
bootstrap_dir = "/Volumes/elmichelangelo_external_ssd_1/Data/Bootstrap6/"
destination_dir = "/Volumes/elmichelangelo_external_ssd_1/Data/Bootstrap6/100"

# Get a list of all subdirectories in the Bootstrap directory
subdirs = [d for d in os.listdir(bootstrap_dir) if os.path.isdir(os.path.join(bootstrap_dir, d))]

# Loop over each subdirectory
for subdir in subdirs:
    # Construct the full path to the h5 file
    # h5_file_path = os.path.join(bootstrap_dir, subdir, "catalogs", "*.h5")
    h5_file_path = glob.glob(os.path.join(bootstrap_dir, subdir, "catalogs", "*.h5"))

    for h5_file in h5_file_path:
        try:
            # Check if the h5 file exists
            if os.path.isfile(h5_file):
                # Copy the h5 file to the destination directory
                shutil.move(h5_file, destination_dir)
        except IndexError:
            print(f"No h5 file found in {subdir}")
            continue
