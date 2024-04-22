import os
import pandas as pd
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt


def print_structure(h5_file, path=''):
    for key in h5_file.keys():
        item = h5_file[key]
        new_path = f'{path}/{key}'
        if isinstance(item, h5py.Dataset):
            print(f'Dataset: {new_path}')
        elif isinstance(item, h5py.Group):
            print(f'Group: {new_path}')
            print_structure(item, new_path)


# Define the path to the Bootstrap folder
bootstrap_path = "/Volumes/elmichelangelo_external_ssd_1/Data/Bootstrap"

# Get a list of all subfolders in the Bootstrap folder
subfolders = [f.path for f in os.scandir(bootstrap_path) if f.is_dir()]

# Initialize an empty DataFrame to store the results

# results = pd.DataFrame()

counter = 1

lst_columns = [
    b"BDF_MAG_DERED_CALIB_R",
    b"unsheared/mag_r",
]

for col in lst_columns:
    results = []

    # Loop over the subfolders
    for subfolder in subfolders:
        # Construct the path to the h5 file in the Catalog subsubfolder
        h5_file_paths = glob.glob(os.path.join(subfolder, "catalogs", "*.h5"))

        with h5py.File(h5_file_paths[0], "r") as h5_file:
            print_structure(h5_file)

        with h5py.File(h5_file_paths[0], "r") as h5_file:
            # Read the datasets from the HDF5 file
            group = h5_file['df']
            axis0 = group['axis0'][:]
            axis1 = group['axis1'][:]
            block0_items = group['block0_items'][:]
            block0_values = group['block0_values'][:]
            block1_items = group['block1_items'][:]
            block1_values = group['block1_values'][:]

            # Create the DataFrame
            df = pd.DataFrame(data=block0_values, columns=block0_items)

            # Calculate the mean of the desired column
            mean = df[col].mean()  # replace "column_name" with the actual column name

            # Append the result to the results DataFrame
            # results = results.append({"BDF_MAG_DERED_CALIB_R": counter, "Mean": mean}, ignore_index=True)
            results.append(mean)
            counter += 1

    plt.figure(figsize=(10, 5))
    plt.plot(results, 'ro')  # 'ro' makes the point a red circle
    plt.hlines(np.mean(results), xmin=0, xmax=len(results), colors='b', linestyles='dashed')
    plt.hlines(np.mean(results)-np.std(results), xmin=0, xmax=len(results), colors='k', linestyles='dashed')
    plt.hlines(np.mean(results)+np.std(results), xmin=0, xmax=len(results), colors='k', linestyles='dashed')
    plt.title(f'{col} Mean Value')
    plt.xlabel('Index')
    plt.ylabel('Mean')
    plt.show()

exit()
# Save the results DataFrame to a CSV file
results.to_csv("results.csv", index=False)