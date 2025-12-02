from PIL import Image
import cv2
import numpy as np
import imageio
import argparse

def args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--obj', type = str, required = True, help='path to the .obj file') 

    parser.add_argument('--face', type = str, required = True, help='path to the face texture')

    parser.add_argument('--name', type = str, required = True, help='subject name')

    arguments = parser.parse_args()
    return arguments


def main():
    arguments = args()
    obj = arguments.obj
    face = arguments.face
    name = arguments.name

    name_mtl = f'{name}_materiale.mtl'

    # Modifica file obj per inserire le info sull'uv map

    """In questo codice, apriamo il file OBJ e leggiamo tutte le sue righe in una lista. 
    Successivamente, apriamo il file di testo contenente il contenuto da sostituire e leggiamo le sue righe in un'altra lista. 
    Poi sostituiamo le righe dalla riga 6892 in poi nella lista delle righe OBJ con la lista di sostituzione. 
    Infine, sovrascriviamo il file OBJ con la nuova lista di righe."""

    with open(obj, 'r') as f:
        lines = f.readlines()

    with open('texture/uv_map.txt', 'r') as f2:
        replacement = f2.readlines()

    lines[10477:] = replacement

    lines[10477] = "mtllib " + name_mtl + "\n"

    lines[10478] = "o " + name + "\n"

    with open(f'texture/texture_{name}.obj', 'w') as f:
        f.writelines(lines)


    # Creazione texture

    # Carica l'immagine di origine
    img = Image.open(face)
    img_width, img_height = img.size

    # Cattura un gruppo di pixel adiacenti
    x, y = 197, 194
    size = 15
    crop_box = (x, y, x+size, y+size)
    crop = img.crop(crop_box)

    # Genera una nuova immagine grande come l'originale e la colora con i valori dei pixel acquisiti
    new_img = Image.new('RGB', (img_width, img_height), (0, 0, 0))

    crop_resized = crop.resize((img_width, img_height))

    # copiamo l'immagine crop ridimensionata nell'immagine più grande
    new_img.paste(crop_resized, (0, 0))
    
    new_img.save(f'texture/{name}_skin.png', 'PNG')

    # Leggi le tre immagini
    img1 = cv2.imread(face, cv2.IMREAD_COLOR)
    img2 = cv2.imread(f'texture/{name}_skin.png', cv2.IMREAD_COLOR)
    mask = cv2.imread('texture/mask.png', cv2.IMREAD_COLOR)
  

    img1 = np.float32(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = np.float32(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    mask = np.float32(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)) / 255.0

    # # effettua il blending
    blended = img1 * mask + img2 * (1 - mask)
    blended = np.uint8(np.clip(blended, 0, 255))
    

    imageio.imwrite(f'texture/{name}_texture.png', blended)

    # Creazione file MTL
    texture = f'texture/{name}_texture.png'

    with open(f'texture/{name}_materiale.mtl', 'w') as f:
        f.write('# Blender MTL File: \'None\'\n')
        f.write('# Material Count: 1\n\n')
        f.write('newmtl Default_OBJ\n')
        f.write('Ns 150.000000\n')
        f.write('Ka 1.000000 1.000000 1.000000\n')
        f.write('Kd 0.800000 0.800000 0.800000\n')
        f.write('Ks 0.500000 0.500000 0.500000\n')
        f.write('Ke 0.000000 0.000000 0.000000\n')
        f.write('Ni 1.450000\n')
        f.write('d 1.000000\n')
        f.write('illum 2\n')
        f.write(f'map_Kd {texture}\n')




if __name__ == '__main__':
    main()