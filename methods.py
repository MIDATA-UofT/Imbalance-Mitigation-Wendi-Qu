import numpy as np
import cv2
import imutils
import random
from torchvision import transforms
from PIL import Image


def horizontal_flip(img):
    return np.fliplr(img)

def gaussian_noise(img):
    row, col, _ = img.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row,col)) 

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return noisy_image.astype(np.uint8)

def rotation(img):
    angle = random.randint(10, 30)
    sign = random.choice([-1,1])
    return imutils.rotate_bound(img, sign*angle)


def affine(img):
    img = Image.fromarray(img)
    img2 = transforms.RandomAffine(0, translate=(0.4, 0.5))(img)
    return np.asarray(img2)


def all_methods(img):
    func = random.choice([horizontal_flip, gaussian_noise, rotation, affine])
    return func(img)