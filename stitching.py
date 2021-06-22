import matplotlib.pyplot as plt
import numpy as np

import util_functions
import cv2

list_image_names = ["images/S1.jpg", "images/S2.jpg", "images/S3.jpg", "images/S5.jpg"]
images = []
for name in list_image_names:
    images.append(cv2.imread(name))

pairwise_stitches = []
while(len(images) > 1):
    while(len(images) >= 2):
        img1 = images[0]
        img2 = images[1]
        stitched = util_functions.stitch(img1, img2)
        pairwise_stitches.append(stitched)

        images.remove(img1)
        images.remove(img2)

    if(len(images) == 1):
        pairwise_stitches.append(images.pop())

    images.clear()
    images.extend(pairwise_stitches)
    pairwise_stitches.clear()

result = images[0]
plt.xlabel("Final panorama", fontsize=14)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()

#result = cv2.resize(result, dsize=(int(result.shape[1] * 2.5), int(result.shape[0] * 2.5)), interpolation=cv2.INTER_CUBIC)
#result = cv2.medianBlur(result, ksize=15)
#result = cv2.resize(result, dsize=(int(result.shape[1] * 0.4), int(result.shape[0] * 0.4)), interpolation=cv2.INTER_CUBIC)
#plt.xlabel("Blurred and scaled down panorama", fontsize=14)
#plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#plt.show()