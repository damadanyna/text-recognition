import cv2
import numpy as np
import pytesseract

# Lire l'image
img = cv2.imread('cmd.png')

# Redimensionner l'image pour améliorer la reconnaissance
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtrage bilatéral pour réduire le bruit tout en conservant les bords
gray = cv2.bilateralFilter(gray, 9, 75, 75)

# Appliquer le seuillage
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optionnel : dilatation et érosion pour améliorer la visibilité des caractères
kernel = np.ones((1, 1), np.uint8)
img_dilated = cv2.dilate(thresh, kernel, iterations=1)
img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

# Trouver les contours des caractères
contours, _ = cv2.findContours(img_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = cv2.drawContours(img_eroded.copy(), contours, -1, (0, 255, 0), 3)

# Configurations personnalisées pour Tesseract
custom_config = r'--oem 3 --psm 6'

# Utiliser pytesseract pour extraire le texte de l'image prétraitée
text = pytesseract.image_to_string(img_contours, config=custom_config)
print(text)

# Afficher l'image traitée avec OpenCV
cv2.imshow('img', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
