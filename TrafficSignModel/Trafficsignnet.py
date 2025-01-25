import tensorflow as tf
from keras import layers, models

# Définir le modèle séquentiel
model = models.Sequential()

# Première couche convolutionnelle
model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 3)))

# Deuxième couche convolutionnelle
model.add(layers.Conv2D(60, (5, 5), activation='relu'))

# Première couche de Max Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Troisième couche convolutionnelle
model.add(layers.Conv2D(30, (3, 3), activation='relu'))

# Quatrième couche convolutionnelle
model.add(layers.Conv2D(30, (3, 3), activation='relu'))

# Deuxième couche de Max Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Dropout pour éviter le surapprentissage
model.add(layers.Dropout(0.5))

# Aplatir la sortie des convolutions pour passer à la couche dense
model.add(layers.Flatten())

# Première couche dense
model.add(layers.Dense(500, activation='relu'))

# Dropout supplémentaire
model.add(layers.Dropout(0.5))

# Couche de sortie avec 43 classes (softmax pour la classification)
model.add(layers.Dense(43, activation='softmax'))

# Afficher le résumé du modèle
model.summary()

# Compiler le modèle
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Entraînement (juste un exemple, à ajuster en fonction de vos données)
# model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
