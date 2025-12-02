import cv2
import numpy as np
import matplotlib.pyplot as plt

def connected_component_label(path):
    
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]   #60
    # Applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)
    # labels = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    from skimage import measure, color
    import pandas as pd

    # compute image properties and return them as a pandas-compatible table
    p = ['label', 'area', 'centroid']
    props = measure.regionprops_table(labels, img, properties=p)
    df = pd.DataFrame(props)
    print(df)

    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()

    centroid0 = df['centroid-0']
    centroid1 = df['centroid-1']

    image = np.ones(img.shape, dtype="uint8")

    rang = num_labels 

    if side == 'front':
        for i in range(rang-1): #145, 180, 170, 185
            #if ((centroid0[int(i)]>145 and centroid0[int(i)]<180 and centroid1[int(i)]>170 and centroid1[int(i)]<185) or (centroid0[int(i)]>95 and centroid0[int(i)]<100 and centroid1[int(i)]>85 and centroid1[int(i)]<95) or (centroid0[int(i)]>95 and centroid0[int(i)]<105 and centroid1[int(i)]>255 and centroid1[int(i)]<270)): 
            if ((centroid1[int(i)]>245 and centroid1[int(i)]<255) or (centroid0[int(i)]>95 and centroid0[int(i)]<100 and centroid1[int(i)]>85 and centroid1[int(i)]<95) or (centroid0[int(i)]>95 and centroid0[int(i)]<105 and centroid1[int(i)]>255 and centroid1[int(i)]<270)): 
                image = np.bitwise_or(image, (labels == int(i+1)).astype("uint8")*255)
    else:
        for i in range(rang-1):
            #if ((centroid0[int(i)]>160 and centroid0[int(i)]<180 and centroid1[int(i)]>165 and centroid1[int(i)]<190) or (centroid0[int(i)]>75 and centroid0[int(i)]<85 and centroid1[int(i)]>175 and centroid1[int(i)]<190) or (centroid0[int(i)]>225 and centroid0[int(i)]<235 and centroid1[int(i)]>175 and centroid1[int(i)]<190)):
            if ((centroid1[int(i)]>245 and centroid1[int(i)]<260) or (centroid0[int(i)]>75 and centroid0[int(i)]<85 and centroid1[int(i)]>175 and centroid1[int(i)]<190) or (centroid0[int(i)]>225 and centroid0[int(i)]<235 and centroid1[int(i)]>175 and centroid1[int(i)]<190)):
                image = np.bitwise_or(image, (labels == int(i+1)).astype("uint8")*255)

    cv2.imshow("mask", image)
    cv2.waitKey(0)

    image = (labels == 1).astype("uint8")*255

    return image

side = 'front'
ret = connected_component_label('D:/Tesi/Moro/3DAvatarGenerator/data4/female_sil/3140/512/side.png')
cv2.imshow("ret", ret)
cv2.waitKey(0)


