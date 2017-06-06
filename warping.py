import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Function To find the keypoints using SURF
def keypoints(image):
    img = cv2.imread(image)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # SUFR Detector
    surf = cv2.SURF(9000)
    
    kp2, d2 = surf.detectAndCompute(gray,None,useProvidedKeypoints = False)

    return [kp2, d2, gray]
######################################################################################################
##    img = cv2.drawKeypoints(gray, kp2,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##    cv2.imwrite('surfImage.jpg', img)
######################################################################################################   

def matching(image1, image2, isprint):
    # Get the keypoints, descriptor and the corresponding grey image for both the images
    kp1, d1, gray1 = keypoints(image1)
    kp2, d2, gray2 = keypoints(image2)

    # using the Brute Force matcher
    matchObj = cv2.BFMatcher(cv2.NORM_L2, False) # default values

    # Call Match function to match the image using descriptors
    matchedObj = matchObj.match(d1, d2)             

    #print "Number of Keypoints for first image is %d" % len(kp1)
    #print "Number of Keypoints for second image is %d" % len(kp2)

    # Thresholding 1: Sorting the distances
    matchedObj = sorted(matchedObj, key = lambda x:x.distance)

    # Thresholding 2: Sum of distance by the length of all multipled by a constant
    dist = [m.distance for m in matchedObj]
    thres_dist = (sum(dist) / len(dist)) * 0.90

    # Selcted only the points that pass the threshold
    matchedObj = [m for m in matchedObj if m.distance < thres_dist]

    if isprint == True: 
        # For plotting
        h1, w1 = gray1.shape[:2]
        h2, w2 = gray2.shape[:2]
        view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
        view[:h1, :w1, 0] = gray1
        view[:h2, w1:, 0] = gray2
        view[:, :, 1] = view[:, :, 0]
        view[:, :, 2] = view[:, :, 0]

        # Thresholding 1: take the top k values
        # for m in matchedObj[:50]:
        for m in matchedObj:
            color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
            cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), cv2.cv.CV_RGB(color[0],color[1],color[2]))
        
        cv2.imwrite('matching.jpg', view)
    else:
        return matchedObj    

# Function to get the Affine Matrix
def affine(image1, image2):

    # Get the keypoints, descriptor and the corresponding grey image for both the images
    kp1, d1, gray1 = keypoints(image1)
    kp2, d2, gray2 = keypoints(image2)

    # Get the matched lines
    matches = matching(image1, image2,False)
    # Sorting the distances
    matches = sorted(matches, key = lambda x:x.distance)

    l1 = []
    l2 = []

    # Create two lists with 3 points in each
    # one for the query image and the other for the train image
    for i in range(3):
        l1.append(kp1[matches[i].queryIdx].pt)
        l2.append(kp2[matches[i].trainIdx].pt)

    # Convert the list to floats
    pts1 = np.float32(l1)
    pts2 = np.float32(l2)
    
    # Get the Affine Transform of the two images
    M = cv2.getAffineTransform(pts1,pts2)
    print M

# Function to remove the black area
def imageCrop(image1):
    threshold = 0
    image = cv2.imread(image1)
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
  #  cv2.imwrite(image1, image)

# Find the Homography matrix and print or warp the images
# Make sure the parameters are image2, image1 and isPrint
def homography(image2, image1, isPrint):
    img = cv2.imread(image1)
    img2 = cv2.imread(image2)
    
    # Get the keypoints, descriptor and the corresponding grey image for both the images
    kp1, d1, gray1 = keypoints(image1)
    kp2, d2, gray2 = keypoints(image2)

    # Get the matched lines
    matches = matching(image1, image2, False)
    # Sorting the distances
    matches = sorted(matches, key = lambda x:x.distance)

    l1 = []
    l2 = []

    # Create two lists with 4 points in each
    # one for the query image and the other for the train image
    for i in range(len(matches)):
        l1.append(kp1[matches[i].queryIdx].pt)
        l2.append(kp2[matches[i].trainIdx].pt)

    # Convert the list to floats
    pts1 = np.float32(l1)
    pts2 = np.float32(l2)

    # cv2.LMEDS, 0
    M, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)

    if isPrint == True:
        print M
    else:
        # Warp the 2nd image
        dst = cv2.warpPerspective(img,M,(img.shape[1] + img2.shape[1],img.shape[0]))
        # add the 1st image
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2

        imgName = 'stitchedImg.jpg'
        cv2.imwrite(imgName, dst)
        # Crop the resultant image and return
        return imageCrop(imgName)

# Function to generate a warping a sequence of images
def panoImage():
    # Array to store the images
    inputList = ['IMG_1203.jpg', 'IMG_1204.jpg', 'IMG_1205.jpg', 'IMG_1206.jpg', 'IMG_1207.jpg']
    two_dash = 'ztwo_dash.jpg'
    three_dash = 'zthree_dash.jpg'
    four_dash = 'zfour_dash.jpg'
    five_dash = 'zfive_dash.jpg'

    # Warping the first and the second image
    dashSec = homography(inputList[0], inputList[1], False)
    cv2.imwrite(two_dash, dashSec)

    # Warping the resultant of (1st & 2nd) with 3rd
    dashThird = homography(two_dash, inputList[2], False)
    cv2.imwrite(three_dash, dashThird)

    # Warping the resultant of (1-2'-3') with 4th image
    dashforth = homography(three_dash, inputList[3], False)
    cv2.imwrite(four_dash, dashforth)

    # Warping the resultant of (1-2'-3'-4') with 5th image
    dashfifth = homography(four_dash, inputList[4], False)
    cv2.imwrite(five_dash, dashfifth)
#-------------------------------------------------------------
# Matching two images
#matching('IMG_1200.jpg', 'IMG_1201.jpg', True)

# Get the Affine Matrix
#affine('IMG_1188.jpg', 'IMG_1189.jpg')

# Get the Homography Matrix
#homography('IMG_1208.jpg', 'IMG_1209.jpg', True)

# Warp Two images and save the output
imgs = homography('IMG_1207.jpg', 'IMG_1206.jpg', False)
cv2.imwrite("warping.jpg",imgs)

# Warp Multiple Images Together
#panoImage()
