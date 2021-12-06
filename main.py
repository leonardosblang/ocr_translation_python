import cv2 
import pytesseract
from deep_translator import GoogleTranslator



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def cleanImage(image):
    imgclean = grayscale(image)
    imgclean = thresholding(imgclean)
    return imgclean

img = cv2.imread('Untitled.png')
imageclean = cleanImage(img)
custom_config = r'--oem 3 --psm 6'
ocr_txt = pytesseract.image_to_string(imageclean, config=custom_config)
print(ocr_txt)

translated = GoogleTranslator(source='auto', target='pt').translate(ocr_txt)
print(translated)

cv2.imshow('img', imageclean)
cv2.waitKey(0)