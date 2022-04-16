import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def load_dataset(path, grid_size, classes, b_boxes, batch):
    images_num = sum(f.endswith("png") for root, dirs, files in os.walk(path) for f in files)

    image_generator = ImageDataGenerator(rescale=1./255)

    x = np.zeros((images_num, 224, 224, 3))
    y = np.zeros((images_num, grid_size, grid_size, b_boxes, 5 + classes))

    print("Loading dataset...")
    index = 0
    with tqdm(total=images_num) as pbar:
        for class_dir in os.listdir(os.path.join(path, "annotations")):
            for file in os.listdir(os.path.join(path, "annotations", class_dir)):
                if file.endswith(".xml"):
                    img_path = os.path.join(path, "images", class_dir, file.replace("xml", "png"))

                    x[index] = img_to_array(load_img(img_path))
                    y[index] = get_object_array(path, class_dir, file, grid_size, classes, b_boxes)
                    index += 1
                    pbar.update(1)

    y = np.asarray(y).astype("float32")
    return image_generator.flow(x, y, batch_size=batch)


def get_object(path, class_dir, file):
    root = ET.parse(os.path.join(path, "annotations", class_dir, file)).getroot()
    child = root.find("object")

    obj = {"class": child.find("name").text}

    bndbox = child.find('bndbox')
    obj["xmax"] = float(bndbox.find("xmax").text)
    obj["xmin"] = float(bndbox.find("xmin").text)
    obj["ymax"] = float(bndbox.find("ymax").text)
    obj["ymin"] = float(bndbox.find("ymin").text)

    return obj


def get_object_array(path, class_dir, file, grid_size, classes, b_boxes):
    class_names = os.listdir(os.path.join(path, "images"))

    obj = get_object(path, class_dir, file)

    array = np.zeros((grid_size, grid_size, b_boxes, 5 + classes))

    confidence = 1

    center_x = (obj["xmin"] + obj["xmax"]) / 2
    center_x = center_x / (224 / grid_size)
    x = center_x - np.floor(center_x)
    grid_x = int(np.floor(center_x))

    center_y = (obj["ymin"] + obj["ymax"]) / 2
    center_y = center_y / (224 / grid_size)
    y = center_y - np.floor(center_y)
    grid_y = int(np.floor(center_y))

    w = (obj["xmax"] - obj["xmin"]) / 224
    h = (obj["ymax"] - obj["ymin"]) / 224

    class_array = [0.] * classes
    class_array[class_names.index(obj["class"])] = 1.

    array[grid_x, grid_y, 0] = [x] + [y] + [w] + [h] + [confidence] + class_array

    return array
