import numpy as np
import cv2

# Automatic Panoramic Image Stitching using Invariant Features - Matthew Brown and David G. Lowe
# http://matthewalunbrown.com/papers/ijcv2007.pdf

# Parameters
m_param = 6
k_param = 4
imagePaths = ['images/bryce_left_01.png', 'images/bryce_left_02.png', 'images/bryce_left_03.png', 'images/bryce_right_01.png', 'images/bryce_right_02.png', 'images/bryce_right_03.png']

descriptor = cv2.SIFT_create()
keypoints = []
descriptors = []

images = []
for path in imagePaths:
	tmpImg = cv2.imread(path)
	images.append(tmpImg)

	tmpImgGray = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
	k, d = descriptor.detectAndCompute(tmpImgGray, None)

	keypoints.append(k)
	descriptors.append(d)

homographies = []
best_matches = []

# TODO: does it HAVE to be O(n^2)
matcher = cv2.FlannBasedMatcher()
for i in range(0, len(images)-1):
	homographies.append([])
	good_matches_list = []

	for j in range(i + 1, len(images)):
		matches = matcher.knnMatch(descriptors[i], descriptors[j], k=2)

		# ratio test
		good_matches = []
		for m, n in matches:
			if(m.distance < 0.75 * n.distance):
				good_matches.append([m])

		good_matches_list.append((descriptors[j], keypoints[j], good_matches))

	good_matches_list = sorted(good_matches_list, key=lambda y: len(y[2]), reverse=True)
	good_matches_list = good_matches_list[:m_param]
	best_matches[i] = good_matches_list[0]

	# Locations of best matches
	k1 = keypoints[i]

	for j in range(len(good_matches_list)):
		match = good_matches_list[j]
		k2 = keypoints

		points1 = np.array([k1[match[0].queryIdx].pt for match in match[2]], dtype=np.float32)
		points1 = points1.reshape((-1, 1, 2))
		points2 = np.array([k2[match[0].trainIdx].pt for match in match[2]], dtype=np.float32)
		points2 = points2.reshape((-1, 1, 2))

		h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5)
		homographies[i][j] = h

# Anchor image: I1
I1 = 1
H1 = [[1, 0, 0],[0, 1, 0], [0, 0, 1]]  # Identity
currentImage = 1
while True:
	currentImageData = best_matches[currentImage]
	homographies