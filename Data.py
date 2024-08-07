import os
import shutil

# Source folder containing 35 subfolders, each with images
source_folder = r"D:\Pycharm\ASL2H\Images"

# Destination folder where you want to move the images
destination_folder = r"D:\Pycharm\ASL2H\Dataset"

# Number of images to take from each subfolder
num_images_to_take = 20

# Iterate through each subfolder in the source folder
for subfolder_name in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder_name)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # List all files in the subfolder
        files_in_subfolder = os.listdir(subfolder_path)

        # Take the first 100 images (or less if there are fewer than 100)
        selected_files = files_in_subfolder[:num_images_to_take]

        # Move each selected file to the destination folder
        for file_name in selected_files:
            source_file_path = os.path.join(subfolder_path, file_name)
            destination_file_path = os.path.join(destination_folder, file_name)

            # Move the file
            shutil.move(source_file_path, destination_file_path)

            print(f"Moved: {file_name} from {subfolder_name} to {destination_folder}")

print("Finished moving images.")
