import cv2
import numpy as np
import pytesseract

# Assurez-vous que ce chemin est correct et correspond à l'emplacement de tesseract.exe sur votre système
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

# Lire l'image
img = cv2.imread('images.png')

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer le seuillage
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optionnel : dilatation ou érosion pour améliorer les caractères
kernel = np.ones((1, 1), np.uint8)
img_dilated = cv2.dilate(thresh, kernel, iterations=1)

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img_dilated, config=custom_config)
# Utiliser pytesseract pour extraire le texte de l'image prétraitée
print(pytesseract.image_to_string(img_dilated))

# Afficher l'image traitée avec OpenCV
cv2.imshow('img', img_dilated)
cv2.waitKey(0)
