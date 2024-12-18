import os
import shutil


def keep_first_n_folders(dataset_path, n=100):
    """
    Keeps only the first 'n' subfolders in the given dataset directory and deletes the rest.

    Args:
        dataset_path (str): The path to the dataset directory.
        n (int): The number of subfolders to keep.

    Returns:
        None
    """
    items = os.listdir(dataset_path)
    folders = [item for item in items if os.path.isdir(
        os.path.join(dataset_path, item))]
    folders.sort()

    if len(folders) <= n:
        print(
            f"The dataset directory contains {len(folders)} folders, which is less than or equal to {n}. No folders will be deleted.")
        return

    folders_to_delete = folders[n:]

    print(f"Total folders found: {len(folders)}")
    print(f"Keeping the first {n} folders.")
    print(f"Deleting {len(folders_to_delete)} folders...")

    for folder_name in folders_to_delete:
        folder_path = os.path.join(dataset_path, folder_name)
        try:
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")

    print("Folder cleanup complete.")


if __name__ == "__main__":
    dataset_directory = 'TODO'
    number_of_folders_to_keep = 100

    keep_first_n_folders(dataset_directory, number_of_folders_to_keep)
