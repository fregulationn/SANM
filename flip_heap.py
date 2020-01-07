import cv2 as cv

from PIL import Image

im = Image.open("/home/junjie/Downloads/addd.jpg")
out = im.transpose(Image.FLIP_TOP_BOTTOM)
out.show()