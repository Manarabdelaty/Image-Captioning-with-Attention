from PIL import Image
import numpy as np
import os
import coco
import matplotlib.pyplot as plt

def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """

    # Load the image using PIL.
    img = Image.open(path)

    # Resize image if desired.
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array.
    img = np.array(img)

    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img

def show_image(filenames, captions,idx, train):
    """
    Load and plot an image from the training- or validation-set
    with the given index.
    """

    if train:
        # Use an image from the training-set.
        dir = coco.train_dir
        filename = filenames[idx]
        captions = captions[idx]
    else:
        # Use an image from the validation-set.
        dir = coco.val_dir
        filename = filenames[idx]
        captions = captions[idx]

    # Path for the image-file.
    path = filename          #os.path.join(dir, filename)

    # Print the captions for this image.
    for caption in captions:
        print(caption)
    
    print(path)
    # Load the image and plot it.
    img = load_image(path)
    plt.imshow(img)
    plt.show()