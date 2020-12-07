# Document-Scanning
Using OpenCV, we get a flattened image of a document kept in front of the camera in real time.  
We take an image from the live feed of the webcam and preprocess it.  
Then we find contours in the image and choose the biggest one as this will be the document .  
We reorder the points of the contours and then get the warped(in this case flattened) image using opencv functions.
Finally we display this image on the output window.
