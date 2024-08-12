import cv2
import os
from typing import Tuple
from multiprocessing import Pool
import tqdm


def resize_and_dump(path, img, size: Tuple[int, int], folder="data"):
    image_resize = cv2.resize(img, size)
    base_folder = f"{folder}/{size[0]}_{size[1]}_crop/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    cv2.imwrite(
        base_folder + path[:-4] + "_crop" + path[-4:],
        image_resize,
    )


def read_crop_face(path: str):
    cascPath = "./data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(os.path.abspath("data/celeba/" + path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 5, 5)

    if len(faces) != 0:
        x, y, w, h = faces[0]
        only_face = image[y : y + w, x : x + w, :]
        resize_and_dump(path, only_face, (32, 32))
        resize_and_dump(path, only_face, (64, 64))
    return path


if __name__ == "__main__":
    files = sorted(os.listdir("data/celeba/"))
    with Pool(15) as p:
        l = list(tqdm.tqdm(p.imap(read_crop_face, files), total=len(files)))
