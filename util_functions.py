import numpy as np
import cv2
import matplotlib.pyplot as plt

def stitch(img1, img2):
    # За наоѓање на дескрипторите и клучните точки, подобро е процесирање
    # врз монохроматска / црно-бела слика, затоа ги конвертираме сликите.
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    result = RANSAC_SIFT(img1, img2, img1_gray, img2_gray)

    # Правиме threshold за да се најдат контурите
    grayResult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(grayResult, 0, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    resContoutes = cv2.drawContours(result.copy(), contours, -1, (0, 255, 255))

    plt.xlabel("Found contours", fontsize=14)
    plt.imshow(cv2.cvtColor(resContoutes, cv2.COLOR_BGR2RGB))
    plt.show()

    # Најди bounding box на контурата која има најголема плоштина
    contour_maxArea = max(contours, key=cv2.contourArea)
    (x, y, width, height) = cv2.boundingRect(contour_maxArea)
    result = result[y: y + height, x: x + width]
    plt.xlabel("Removed rightmost empty black space", fontsize=14)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

    return result

def resize(img, width = None, height = None):
    if(width == None and height == None):
        return img

    if(width == None):
        ratio = float(height) / float(img.shape[0])
        newWidth = int(ratio * img.shape[1])
        return cv2.resize(img, (newWidth, height))

    else:
        ratio = float(width) / float(img.shape[1])
        newHeight = int(ratio * img.shape[0])
        return cv2.resize(img, (width, newHeight))

def RANSAC_SIFT(img1, img2, img1_gray, img2_gray):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))

    ax1.set_xlabel("Image 1", fontsize=14)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    ax2.set_xlabel("Image 2", fontsize=14)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

    descriptorDetector = cv2.SIFT_create()
    k1, d1 = descriptorDetector.detectAndCompute(img1_gray, None)
    k2, d2 = descriptorDetector.detectAndCompute(img2_gray, None)

    # # Brute force matcher
    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(d1, d2, k=2)
    #
    # # Ratio test
    # good_matches = []
    # for m, n in matches:
    #     if(m.distance < 0.75 * n.distance):
    #         good_matches.append([m])

    # Brute force matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
    matches = matcher.match(d1, d2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # The best X%
    num_good_matches = int(len(matches) * 0.1)
    good_matches = matches[:num_good_matches]

    if(len(good_matches) < 4):
        print("ERROR! TOO FEW POINTS FOR HOMOGRAPHY!")

    imMatches = cv2.drawMatches(img1, k1, img2, k2, good_matches, None, flags=2)
    plt.xlabel("Found matches", fontsize=14)
    plt.imshow(cv2.cvtColor(imMatches, cv2.COLOR_BGR2RGB))
    plt.show()

    # Locations of best matches
    points1 = np.array([k1[match.queryIdx].pt for match in good_matches], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = np.array([k2[match.trainIdx].pt for match in good_matches], dtype=np.float32)
    points2 = points2.reshape((-1, 1, 2))

    # Compute homography (2to1)
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp image plane using homography
    result2to1 = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
    plt.xlabel("Warped Image2 to Image1's plane", fontsize=14)
    plt.imshow(cv2.cvtColor(result2to1, cv2.COLOR_BGR2RGB))
    plt.show()

    # Stitch img1 and img2 together
    result2to1[0:img1.shape[0], 0:img1.shape[1]] = img1
    plt.xlabel("Resulting stitched image (img2 to img1)", fontsize=14)
    plt.imshow(cv2.cvtColor(result2to1, cv2.COLOR_BGR2RGB))
    plt.show()

    return result2to1

def RANSAC_BRISK(img1, img2, img1_gray, img2_gray):
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)

    descriptorDetector = cv2.BRISK_create()
    k1, d1 = descriptorDetector.detectAndCompute(img1_gray, None)
    k2, d2 = descriptorDetector.detectAndCompute(img2_gray, None)

    # Brute force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(d1, d2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if (m.distance < 0.75 * n.distance):
            good_matches.append([m])

    imMatches = cv2.drawMatchesKnn(img1, k1, img2, k2, good_matches, None, flags=2, matchColor=(0, 0, 255))
    cv2.imshow("Found matches", imMatches)

    # Locations of best matches
    points1 = np.array([k1[match[0].queryIdx].pt for match in good_matches], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = np.array([k2[match[0].trainIdx].pt for match in good_matches], dtype=np.float32)
    points2 = points2.reshape((-1, 1, 2))

    # Compute homography (2to1)
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp image plane using homography
    result2to1 = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    cv2.imshow("Warped img2 to img1's plane of view", result2to1)

    # Stitch img1 and img2 together
    result2to1[0:img1.shape[0], 0:img1.shape[1]] = img1
    cv2.imshow("Resulting stitched image (2to1)", result2to1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# NOTE: потребна е верзијата <=3.4.2 на пакетот opencv-contrib-python
# (Јас ја компајлирав од source code бидејќи pip не поддржува толку стара верзија)
def RANSAC_SURF(img1, img2, img1_gray, img2_gray):
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)

    descriptorDetector = cv2.xfeatures2d.SURF_create()
    k1, d1 = descriptorDetector.detectAndCompute(img1_gray, None)
    k2, d2 = descriptorDetector.detectAndCompute(img2_gray, None)

    # Brute force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(d1, d2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if (m.distance < 0.75 * n.distance):
            good_matches.append([m])

    imMatches = cv2.drawMatchesKnn(img1, k1, img2, k2, good_matches, None, flags=2, matchColor=(0, 0, 255))
    cv2.imshow("Found matches", imMatches)

    # Locations of best matches
    points1 = np.array([k1[match[0].queryIdx].pt for match in good_matches], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = np.array([k2[match[0].trainIdx].pt for match in good_matches], dtype=np.float32)
    points2 = points2.reshape((-1, 1, 2))

    # Compute homography (2to1)
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp image plane using homography
    result2to1 = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    cv2.imshow("Warped img2 to img1's plane of view", result2to1)

    # Stitch img1 and img2 together
    result2to1[0:img1.shape[0], 0:img1.shape[1]] = img1
    cv2.imshow("Resulting stitched image (2to1)", result2to1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()