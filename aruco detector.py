import numpy as np
import cv2
from cv2 import aruco
from dataExtractor import predicting
from dataExtractor import findGraduation
from dataExtractor import arucoPerspective
from matplotlib import pyplot as plt

dict_aruco_form = {34: 0, 35: 1, 36: 2, 33: 3}

dict_form_box = {
    "sid": [(27, 216), (362, 252)],
    "name": [(27, 274), (362, 310)],
    "fname": [(27, 330), (362, 366)],
    "phd": [(47, 396), (61, 412)],
    "mst": [(141, 396), (155, 412)],
    "bch": [(282, 396), (296, 412)],
}

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
dest_points = np.array([(0, 0), (500, 0),
                        (500, 660), (0, 660)])


def preprocess_photo(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    src_points = np.zeros((4, 2), dtype=np.dtype('int32'))

    if len(ids) != 4:
        return None

    for i in range(4):
        aruco_id = ids[i][0]
        corn = dict_aruco_form[aruco_id]
        src_points[corn][0] = corners[i][0][corn][0]
        src_points[corn][1] = corners[i][0][corn][1]

    # calculate Homography and warp image
    H, _ = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 4.0)
    warped = cv2.warpPerspective(img, H, (500, 660))
    return warped


def extract_box(image, point_top_left, point_bottom_right, pieces):
    x_left = point_top_left[0]
    x_right = point_bottom_right[0]
    y_top = point_top_left[1]
    y_bottom = point_bottom_right[1]
    box = image[y_top:y_bottom, x_left:x_right]
    cut = np.array_split(box, pieces, axis=1)
    cut = [cv2.resize(i, (28, 28)) for i in cut]
    return cut


def extract_features(fname):
    warped = preprocess_photo(fname)
    sid = extract_box(warped, dict_form_box["sid"][0], dict_form_box["sid"][1], 8)
    name = extract_box(warped, dict_form_box["name"][0], dict_form_box["name"][1], 8)
    fname = extract_box(warped, dict_form_box["fname"][0], dict_form_box["fname"][1], 8)
    graduation = []
    graduation.append(extract_box(warped, dict_form_box["bch"][0], dict_form_box["bch"][1], 1)[0])
    graduation.append(extract_box(warped, dict_form_box["mst"][0], dict_form_box["mst"][1], 1)[0])
    graduation.append(extract_box(warped, dict_form_box["phd"][0], dict_form_box["phd"][1], 1)[0])
    return (sid, name, fname, graduation)

dir = "Test_set/From (10).jpg"

number, name, family_name, graduation = extract_features(dir)
image = cv2.imread(dir)
cv2.imshow("im" , image)

cv2.imshow("im1" , family_name[0])
cv2.imshow("im2" , family_name[1])
cv2.imshow("im3" , family_name[2])
cv2.imshow("im4" , family_name[3])
cv2.imshow("im5" , family_name[4])
cv2.imshow("im6" , family_name[5])
cv2.imshow("im7" , family_name[6])
cv2.imshow("im8" , family_name[7])
cv2.imshow("im" , image)


numberStr = ""
nameStr = ""
familyNameStr = ""
for k in range(7):
    numberStr += predicting(number[k+1], True, False)

for k in range(8):
    nameStr += predicting(name[k], False, True)

for k in range(8):
    familyNameStr += predicting(family_name[k], False, True)

strr = findGraduation(graduation)
graduationStr = strr

print("number: ",numberStr)
nameStr = nameStr[::-1]
print("name: ",nameStr)
familyNameStr = familyNameStr[::-1]
print("family name: ",familyNameStr)
print("graduation: ",graduationStr)

cv2.waitKey()

# image = cv2.imread("rawData/9321073_1a.JPG")
#
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# parameters =  aruco.DetectorParameters_create()
# corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
# frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
#
# arucoPerspective(image)
#
# print(corners)
# print(ids)
# corners = np.asarray(corners)
# print(corners[0,0,1])
#
# plt.figure()
# plt.imshow(frame_markers)
# for i in range(len(ids)):
#     c = corners[i][0]
#     plt.plot([c[0, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
# plt.legend()
# plt.show()



