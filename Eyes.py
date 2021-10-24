import cv2

# Reconhecimento de olhos
searchEyes = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Importando imagem
image = cv2.imread('images/imagem1.jpg')

# Deixando imagem cinza
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eyes = searchEyes.detectMultiScale(imageGray, scaleFactor=1.05)

# Mostrando as matrizes
print(eyes)

for(x, y, l, a) in eyes:
    # Criando retângulo definindo as propriedades (eixo x e y, cor e largura da borda)
    search = cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)
    # Mapeando as posições
    eyePosition = search[y:y + a, x:x + l]
    eyePositionGray = cv2.cvtColor(eyePosition, cv2.COLOR_BGR2GRAY)
    detected = searchEyes.detectMultiScale(eyePositionGray)

    for(ox, oy, ol, oa) in detected:
        cv2.retangle(eyePosition, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2 )

# Mostrando a imagem em popup
cv2.imshow("Detecta  olhos", image)
cv2.waitKey()