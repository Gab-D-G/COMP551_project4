import os
import pickle
import numpy as np
from skimage.transform import resize
from PIL import Image

image_dir='data/allendata/'
dir_list=os.listdir(image_dir)

filename='data/GO_terms.pkl'
with open(filename, 'rb') as handle:
    GO_data=pickle.load(handle)

i=0
data_list=[]
for gene_id in dir_list:
    try:
        GO_terms=GO_data[str(gene_id)]
        gene_dir=image_dir+gene_id+'/'
        jpg_files=os.listdir(gene_dir)
        for jpg_file in jpg_files:
            jpg_array=np.asarray(Image.open(gene_dir+jpg_file).convert('LA'))
            img=resize(jpg_array, (140,300), anti_aliasing=True) #resize all images to 140x300
            img=img*255 #multiply by 255 to scale images back to the same integer value distribution
            img=np.asarray(img, dtype='uint8')
            data_list.append([GO_terms, img])
    except:
        continue
    i+=1
    if i%10==0:
        break
        print(i)

filename='data.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(data_list, handle)

