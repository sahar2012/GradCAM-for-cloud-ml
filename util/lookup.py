#
# Cloud for ML Final Project
# Cole Smith
# lookup.py
#

import os


# noinspection DuplicatedCode
def find_class_by_file_name(filename_or_path):
    """
    Given a path or a file name, find the class
    name and index of the image.

    :param filename_or_path: String containing path or name
    :return: class name, class index
    """
    # Isolate the file name
    filename = os.path.basename(filename_or_path)
    filename = os.path.splitext(filename)[0]

    # The images within ImageNet classes also have
    # an attached index. Remove it if it exists.
    if '_' in filename:
        filename = filename.split("_")[0]

    with open("data/class.map", 'r') as fp:
        for line in fp:
            line = line.split()
            fi = line[0]
            ind = line[1]
            cls = line[2]

            if fi.lower() == filename.lower():
                return cls, int(ind)

    # No class found with file name
    print("[ WRN ] Could not find a class for:", filename)
    return None, None


# noinspection DuplicatedCode
def find_file_name_by_class(class_name):
    """
    Given a class, find the file path id and index
    for the images.

    :param class_name: String containing name
    :return: file path id, class index
    """
    with open("data/class.map", 'r') as fp:
        for line in fp:
            line = line.split()
            fi = line[0]
            ind = line[1]
            cls = line[2]

            if class_name.lower() == cls.lower():
                return fi, int(ind)

    # No class found with file name
    print("[ WRN ] Could not find a file path for:", class_name)
    return None, None
