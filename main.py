import cv2 
import pytesseract
from deep_translator import GoogleTranslator
from pytesseract import Output


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# "limpando a imagem para escanea-la
def cleanImage(image):
    imgclean = grayscale(image)
    imgclean = thresholding(imgclean)
    return imgclean


#lendo imagem e limpando ela
img = cv2.imread('Untitled.png')
imageclean = cleanImage(img)

#configurando tesseract e scaneando imagem com ele
custom_config = r'--oem 3 --psm 6'
ocr_txt = pytesseract.image_to_string(imageclean, config=custom_config)
print(ocr_txt)


# traduzindo imagem scaneada
translated = GoogleTranslator(source='auto', target='pt').translate(ocr_txt)
print(translated)


# mostra imagem após limpeza
cv2.imshow('img', imageclean)
cv2.waitKey(0)

# mostra o que o pytesseract está detectando desenhando um retangulo (usar para fins de debugging)
d = pytesseract.image_to_data(imageclean, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(imageclean, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', imageclean)
cv2.waitKey(0)