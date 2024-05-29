from PIL import Image
import numpy as np

path="/home/diffusion_models3/semantic-diffusion-model/data/ade20k/images/training/zemani_dic_msk_010.png"

# Charger une image en niveaux de gris
gray_image = Image.open(path).convert('L')

# Convertir l'image en RGB
rgb_image = gray_image.convert('RGB')

# Convertir l'image RGB en un tableau numpy
rgb_array = np.array(rgb_image)

# Extraire les canaux R, G, et B
r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]

# Vérifier si les canaux R, G et B sont identiques
if np.array_equal(r, g) and np.array_equal(g, b):
    print("Les canaux R, G et B sont identiques.")
else:
    print("Les canaux R, G et B ne sont pas identiques.")
