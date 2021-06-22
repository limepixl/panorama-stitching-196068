import matplotlib.pyplot as plt
import util_functions
import cv2

list_image_names = ["images/1.jpg", "images/2.jpg", "images/3.jpg"]
images = []
for name in list_image_names:
    images.append(cv2.imread(name))

pairwise_stitches = []
while(len(images) > 1):
    while(len(images) >= 2):
        img1 = images[0]
        img2 = images[1]
        pairwise_stitches.append(util_functions.stitch(img1, img2))

        images.remove(img1)
        images.remove(img2)

    if(len(images) == 1):
        pairwise_stitches.append(images.pop())

    images.clear()
    images.extend(pairwise_stitches)
    pairwise_stitches.clear()

plt.xlabel("Final panorama", fontsize=14)
plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
plt.show()