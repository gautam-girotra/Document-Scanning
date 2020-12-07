import cv2
import numpy as np

img_width = 640
img_height = 400

webcam_feed = cv2.VideoCapture(0)
webcam_feed.set(10, 150)


def preprocessing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5,5))
    img_dialate = cv2.dilate(img_canny, kernel, iterations=2 )
    img_thres = cv2.erode(img_dialate, kernel, iterations=2)
    return img_thres

def get_contours(img):
    max_area = 0
    biggest_contour = np.array([])
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in  contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area>max_area and len(approx) == 4:
                max_area=area
                biggest_contour=approx
    cv2.drawContours(img_contour, biggest_contour, -1, (255, 0, 0), 3)
    return biggest_contour


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # Shape is (4, 1, 2), We are removing redundant 1
    newPoints = np.zeros((4,1,2),np.int32) # (4,1,2) because we need to return same shape
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[-1] = myPoints[np.argmax(add)]
    sub = np.diff(myPoints,1) # There is array.sum() but no array.diff()
    newPoints[1] = myPoints[np.argmin(sub)] # np.diff does right value - left value so for (width,0) it will be 0-width= -width so lowest value
    newPoints[2] = myPoints[np.argmax(sub)]
    return newPoints


def getWarp(img,biggest):
    #print(biggest.shape)
    biggest=reorder(biggest)
    pts1 = np.float32(biggest) # If we send it without reordering we get error because live feed points will not be in the form that we have defined in the next line
    pts2 = np.float32([[0, 0],[img_width, 0],[0, img_height],[img_width, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Getting transform matrix
    warped_image = cv2.warpPerspective(img, matrix, (img_width, img_height))
    return warped_image

while True:
    success, img = webcam_feed.read()
    img = cv2.resize(img,(img_width, img_height))
    img_contour = img.copy()
    img_thres = preprocessing(img)
    biggest = get_contours(img_thres)

    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        cv2.imshow("ImageWarped", imgWarped)
    else:
        cv2.imshow("Image_Contour",img_contour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


