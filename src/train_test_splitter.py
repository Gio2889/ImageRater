import os
import json
import random
import shutil


def split_image_sets(main_path, train_ratio=0.8, val_ratio=0.15, output_dir="output"):
    """Splits images into training, validation, and holdout sets based on the provided ratios.

    Args:
        json_file_path (str): Path to the JSON file containing image ratings.
        train_ratio (float, optional): Proportion of the dataset to use for training. Defaults to 0.8.
        val_ratio (float, optional): Proportion of the dataset to use for validation. Defaults to 0.15.
        output_dir (str, optional): Directory to output the split datasets. Defaults to 'output'.

    Returns:
        None
    """
    json_file_path = f"{main_path}scores.json"
    images_path = f"{main_path}images/"
    with open(json_file_path, "r") as file:
        data = json.load(file)

    images = list(data.keys())
    random.shuffle(images)
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_set = {img: data[img] for img in images[:train_end]}
    val_set = {img: data[img] for img in images[train_end:val_end]}
    hold_out_set = {img: data[img] for img in images[val_end:]}

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train_set.json"), "w") as train_file:
        json.dump(train_set, train_file)

    with open(os.path.join(output_dir, "val_set.json"), "w") as val_file:
        json.dump(val_set, val_file)

    with open(os.path.join(output_dir, "hold_out_set.json"), "w") as holdout_file:
        json.dump(hold_out_set, holdout_file)

    for img in train_set:
        shutil.copy(
            f"{images_path}{img}",
            os.path.join(output_dir, "train", os.path.basename(img)),
        )
    for img in val_set:
        shutil.copy(
            f"{images_path}{img}",
            os.path.join(output_dir, "val", os.path.basename(img)),
        )
    for img in hold_out_set:
        shutil.copy(
            f"{images_path}{img}",
            os.path.join(output_dir, "holdout", os.path.basename(img)),
        )
    print("--- Dataset created succesfully")


def clean_ratings_and_pictures(main_path):
    """Cleans ratings and corresponding pictures based on the provided JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing image ratings.
        pictures_directory (str): Directory containing the images to clean.

    Returns:
        None
    """

    json_file_path = f"{main_path}scores.json"
    images_path = f"{main_path}images"

    with open(json_file_path, "r") as file:
        data = json.load(file)

    ids_to_remove = []

    for id, rating in data.items():
        if rating == "-":
            ids_to_remove.append(id)
        if not os.path.exists(f"{images_path}/{id}"):
            print(f"{id} not found in the images dir")
            ids_to_remove.append(id)

    for id in ids_to_remove:
        del data[id]
        if os.path.isfile(images_path):
            os.remove(images_path)
            break

    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main_path = "Some_path"
    # clean images of unsuable formats
    clean_ratings_and_pictures(main_path)
    split_image_sets(
        main_path, train_ratio=0.85, val_ratio=0.1, output_dir=f"{main_path}sets"
    )
