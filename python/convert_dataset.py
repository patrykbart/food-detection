import os
import argparse
from PIL import Image
from tqdm import tqdm
from xml.dom import minidom
import xml.etree.cElementTree as ET


def main(args):

    # Create destination directory
    os.mkdir(args.dest)

    # Get class names
    class_names = {}
    for line in open(os.path.join(args.source, "category.txt"), "r").readlines()[1:]:
        line = line.rstrip("\n").split("\t")

        class_names[line[0]] = line[1]

    # Create directory for images and annotations
    os.mkdir(os.path.join(args.dest, "images"))
    os.mkdir(os.path.join(args.dest, "annotations"))

    print("Converting dataset...")
    images_num = sum(f.endswith("jpg") for root, dirs, files in os.walk(args.source) for f in files)
    with tqdm(total=images_num) as pbar:
        for index, class_name in class_names.items():

            # Create directory for every class
            os.mkdir(os.path.join(args.dest, "images", class_name))
            os.mkdir(os.path.join(args.dest, "annotations", class_name))

            # Copy images
            for file in os.listdir(os.path.join(args.source, index)):
                if file.endswith(".jpg"):
                    # Resize image to (224, 224, 3) and copy
                    image = Image.open(os.path.join(args.source, index, file))
                    image = image.resize((224, 224), Image.ANTIALIAS)
                    image.save(
                        os.path.join(args.dest, "images", class_name, f"{file.split('.')[0]}.png"),
                        quality=95
                    )

                    pbar.update(0.5)

            # Create annotations
            for line in open(os.path.join(args.source, index, "bb_info.txt"), "r").readlines()[1:]:
                image_index, xmin, ymin, xmax, ymax = line.split()

                # Normalize bounding box coordinates
                img_width, img_height = Image.open(os.path.join(args.source, index, f"{image_index}.jpg")).size
                xmin = int(int(xmin) * (224 / img_width))
                xmax = int(int(xmax) * (224 / img_width))
                ymin = int(int(ymin) * (224 / img_height))
                ymax = int(int(ymax) * (224 / img_height))

                # Generate XML
                xml = generate_xml(
                    path=os.path.join(os.path.join(args.dest, "images", class_name, f"{image_index}.png")),
                    index=image_index,
                    class_name=class_name,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax
                )

                # Save XML
                xml = minidom.parseString(ET.tostring(xml)).toprettyxml(indent="\t")
                open(os.path.join(args.dest, "annotations", class_name, f"{image_index}.xml"), "w").write(xml)

                pbar.update(0.5)


def generate_xml(path, index, class_name, xmin, ymin, xmax, ymax):
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = class_name
    ET.SubElement(root, "filename").text = f"{index}.png"
    ET.SubElement(root, "path").text = path

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "224"
    ET.SubElement(size, "height").text = "224"
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    object = ET.SubElement(root, "object")
    ET.SubElement(object, "name").text = class_name
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"

    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)

    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="base_dataset")
    parser.add_argument("--dest", type=str, default="dataset")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print("Destination directory already exists!")
        exit(1)

    if os.path.exists(args.dest):
        print("Destination directory already exists!")
        exit(1)

    main(args)
