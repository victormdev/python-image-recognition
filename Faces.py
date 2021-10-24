import cv2

# Carregando xml responsável pela identidicação da função/imagem
loadScript = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Importando imagem
image = cv2.imread('images/imagem1.jpg')

# Deixando imagem cinza
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = loadScript.detectMultiScale(imageGray, scaleFactor=1.05)

# Mostrando as matrizes
print(faces)

for(x, y, l, a) in faces:
    # Criando retângulo definindo as propriedades (eixo x e y, cor e largura da borda)
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Mostrando a imagem em popup
cv2.imshow("Faces", image)

cv2.waitKey()