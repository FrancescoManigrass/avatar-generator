#questo script prende una immagine in input, aggiunge delle barre laterali bianche per renderla quadrata e poi la ridimensiona a 512x512px

from PIL import Image
import argparse

def args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type = str, required = True, help='path to the .png file') 

    arguments = parser.parse_args()
    return arguments



def square_and_resize_image(image_path):
    # Apriamo l'immagine
    with Image.open(image_path) as im:
        # Otteniamo le dimensioni dell'immagine originale
        width, height = im.size
        
        
        # # Calcoliamo la lunghezza della barra laterale bianca necessaria per rendere l'immagine quadrata
        # padding_length = abs(width - height) // 2
        # # Aggiungiamo le barre laterali bianche all'immagine originale per renderla quadrata
        # if width > height:
        #     padding = (padding_length, 0, padding_length + height, height)
        # else:
        #     padding = (0, padding_length, width, padding_length + width)
        # #im = im.crop(padding).copy()
        
        if int(height)>int(width):
            # Determina la dimensione del lato quadrato
            square_size = max(width, height)
            
            # Crea una nuova immagine quadrata con sfondo bianco
            square_image = Image.new('RGB', (square_size, square_size), (255, 255, 255))

            # Calcola la posizione centrale dell'immagine originale nella nuova immagine
            x_offset = (square_size - width) // 2
            y_offset = (square_size - height) // 2

            # Aggiungi l'immagine originale al centro della nuova immagine
            square_image.paste(im, (x_offset, y_offset))

            #square_image.save("D:/Tesi/Moro/output.png")

        else:
            # Calcola le coordinate del rettangolo di ritaglio
            left = (width - height) // 2
            top = (height - height) // 2
            right = left + height
            bottom = top + height

            # Ritaglia l'immagine
            square_image = im.crop((left, top, right, bottom))
        
        # Ridimensioniamo l'immagine quadrata a 512x512 pixel
        square_image = square_image.resize((512, 512), resample=3)

        # Salviamo l'immagine ridimensionata
        square_image.save("D:/Tesi/Moro/output.png")


def main():
    arguments = args()

    path = arguments.path
    
    square_and_resize_image(path)


if __name__ == '__main__':
    main()