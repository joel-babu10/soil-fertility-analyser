from datasets import load_dataset
from pathlib import Path

# Load the dataset from Hugging Face
dataset = load_dataset(
    "Project-AgML/tomato_leaf_disease"
)

# Display the dataset information
print(dataset)
print(dataset["train"].features)

# Create the output directory
output_dir = Path("tomato_external_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

# Download and save every image in its class folder
for index, sample in enumerate(dataset["train"]):

    image = sample["image"]
    label = sample["label"]

    # Convert numerical label into class name
    class_name = dataset["train"].features["label"].int2str(label)

    # Create a folder for each disease
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = class_dir / f"image_{index:05d}.jpg"
    image.convert("RGB").save(image_path)

    if (index + 1) % 500 == 0:
        print(f"Saved {index + 1} images")

print("Dataset downloaded and saved successfully!")