'''
preprocess.py

Reads raw MRI images (grayscale), resizes them to 224×224, converts to RGB by duplicating the channel,
and saves processed images to data/processed/{yes,no}.
'''
import os # read / write data
import cv2 # functions for image processing

# define paths relative to script
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
# target image dimensions for resizing (width, height)
IMG_SIZE = (224, 224)

def ensure_dir(path: str):
    """
    Create the directory if it does not exist.
    Ensures output paths are available for saving processed images.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def preprocess_and_save():
    """
    Process all images from RAW_DIR by:
      1. Loading as grayscale
      2. Resizing to IMG_SIZE
      3. Converting to 3-channel RGB
      4. Saving under PROCESSED_DIR preserving class subfolders
    """
    # ensure output class directories exist
    for cls in ("yes", "no"):
        out_dir = os.path.join(PROCESSED_DIR, cls)
        ensure_dir(out_dir)
        
    # iterate over each class label
    for cls in ("yes", "no"):
        raw_class_dir = os.path.join(RAW_DIR, cls)       # folder with raw images
        proc_class_dir = os.path.join(PROCESSED_DIR, cls) # folder to save processed images

        # loop through all files in the raw class directory
        for fname in os.listdir(raw_class_dir):
            raw_path = os.path.join(raw_class_dir, fname)

            # load the image in grayscale mode (single channel)
            img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Could not read {raw_path}, skipping.")
                continue
            # resize image to the target dimensions
            # INTER_AREA is good for downsampling
            img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

             # convert grayscale to RGB by duplicating channels
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # construct the full save path and write the image
            save_path = os.path.join(proc_class_dir, fname)
            cv2.imwrite(save_path, img_rgb)

        print("✅ Preprocessing complete: images saved to data/processed/")


if __name__ == "__main__":
    """
    When executed as a script, run the preprocessing pipeline.
    Allows: $ python preprocess.py
    """
    preprocess_and_save()

        

