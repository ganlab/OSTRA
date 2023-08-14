import os
import imageio
import numpy as np
from PIL import Image

masked_pred_list = []
fps = 10

for root, dirs, files in os.walk("/media/root/Elements/doc/output/doc/images"):
    for file in files:
        file_path = os.path.join(root, file)
        image = Image.open(file_path)
        arr = np.array(image)
        masked_pred_list.append(arr)

imageio.mimsave("result.gif", masked_pred_list, duration=(1000*1/fps))#update xujx
