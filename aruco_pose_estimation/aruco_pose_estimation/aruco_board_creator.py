import cv2
import numpy as np
import cv2.aruco as aruco


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_id = 5
marker_size = 50

marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_image)

cv2.imshow(f'ArUco Marker ID {marker_id}', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()