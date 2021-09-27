import os
import cv2
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Flatten , Dropout
import numpy as np
from keras.models import Sequential
import time
from cv2 import aruco
from keras.preprocessing.image import load_img, img_to_array

# {'ا': 0, 'ب': 1, 'ت': 2, 'ث': 3, 'ج': 4, 'ح': 5, 'خ': 6, 'د': 7, 'ذ': 8
#     , 'ر': 9, 'ز': 10, 'س': 11, 'ش': 12, 'ص': 13, 'ض': 14, 'ط': 15, 'ظ': 16
#     , 'ع': 17, 'غ': 18, 'ف': 19, 'ق': 20, 'ل': 21, 'م': 22, 'ن': 23, 'ه': 24
#     , 'و': 25, 'پ': 26, 'چ': 27, 'ژ': 28, 'ک': 29, 'گ': 30, 'ی': 31}
classes = ['0','1','2','3','4','5','6','7','8','9' ,
                'ا','ب','پ','ت','ث','ج','چ','ح','خ','د' ,
                'ذ','ر','ز','ژ','س','ش','ص','ض','ط','ظ','ع',
                'غ','ف','ق','ک','گ','ل','م','ن','و','ه' ,
                'ی']

classes_adad = ['0','1','2','3','4','5','6','7','8','9',' ']

classes_horof = [' ' , 'ا','ب','ت','ث','ج','ح','خ','د','ذ' ,'ر','ز','س',
                'ش','ص','ض','ط','ظ','ع','غ','ف','ق','ل','م',
                'ن','ه','و','پ' ,'چ' ,'ژ','ک','گ','ی',]

img_rows = 28
img_cols = 28

def testAruco(image):
    # Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    return len(markerCorners)

def imageResize(I,width,height):
    dim = (width, height)
    resized = cv2.resize(I, dim, interpolation=cv2.INTER_AREA)
    return resized

def perspective(I2,I1):
    sift = cv2.xfeatures2d.SIFT_create()  # opencv 3

    G2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    keypoints2, desc2 = sift.detectAndCompute(G2, None)  # opencv 3

    G1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # distance ratio test
    alpha = 0.60
    good_matches = [m1 for m1, m2 in matches if m1.distance < alpha * m2.distance]

    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1, dtype=np.float32)

    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2, dtype=np.float32)

    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # H = np.eye(3,dtype=np.float32) # this needs to be changed
    pts = np.float32([[0, 0],
                      [G2.shape[1], 0],
                      [G2.shape[1], G2.shape[0]],
                      [0, G2.shape[0]]]).reshape(-1, 1, 2)  # this needs to be changed

    dst = cv2.perspectiveTransform(pts, H).reshape(4, 2)

    J = I1.copy()
    cv2.line(J, (dst[0, 0], dst[0, 1]), (dst[1, 0], dst[1, 1]), (255, 0, 0), 3)
    cv2.line(J, (dst[1, 0], dst[1, 1]), (dst[2, 0], dst[2, 1]), (255, 0, 0), 3)
    cv2.line(J, (dst[2, 0], dst[2, 1]), (dst[3, 0], dst[3, 1]), (255, 0, 0), 3)
    cv2.line(J, (dst[3, 0], dst[3, 1]), (dst[0, 0], dst[0, 1]), (255, 0, 0), 3)

    I = cv2.drawMatches(J, keypoints1, I2, keypoints2, good_matches, None)

    ########transform image
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    output_size = (I2.shape[1], I2.shape[0])
    transformedImage = cv2.warpPerspective(I1, H, output_size)



    # cv2.imshow('keypoints', imageResize(I, 1008, 756))
    # cv2.imshow('keypoints2', imageResize(transformedImage, 1280, 720))

    return imageResize(transformedImage, 1654, 1166)

def arucoPerspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners = np.asarray(corners)

    if len(ids) == 4:
        for i in range(4):
            if ids[i] == 30:
                p1 = corners[i,0,0]
            if ids[i] == 31:
                p2 = corners[i,0,1]
            if ids[i] == 32:
                p3 = corners[i,0,3]
            if ids[i] == 33:
                p4 = corners[i,0,2]

        points1 = np.array([p1, p2, p4, p3], dtype=np.float32)
        # print(points1)

        n = 1654
        m = 1166
        output_size = (n, m)

        p1 = (0, 0)
        p2 = (n, 0)
        p3 = (n, m)
        p4 = (0, m)

        points2 = np.array([p1, p2, p3, p4], dtype=np.float32)

        H = cv2.getPerspectiveTransform(points1, points2)

        J = cv2.warpPerspective(image, H, output_size)

        return J

        # cv2.imshow("im" , J)
        # cv2.waitKey()

    else:
        print("some Aruco didnt detected!")

def extract(pathSampleForms, pathWrittenForms , isForm):
    I2 = cv2.imread(pathSampleForms)
    fnames = os.listdir(pathWrittenForms)
    fnames.sort()
    picNum = 0

    for fname in fnames:
        tic = time.perf_counter()

        id = fname[8:10]
        ####extract step 2
        print("extracting picture number " , picNum  , "of 409" , "id: " , id ,  " completed: " , picNum/409)
        I1 = cv2.imread(pathWrittenForms + "/" + fname)

        # transformedImage = perspective(I2 , I1)

        ####extract step 1
        # if testAruco(I1) == 4:
        # print(fname)
        # cv2.imwrite("masalanDorost/" + fname, transformedImage)

        if (isForm == True):
            print("***********************************************************")
            print("testing Written Forms")
            print("form Name: ",fname)
            print()
            # extractFormImage(transformedImage)
            toc = time.perf_counter()
            print(f"Job done in {toc - tic:0.4f} seconds")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            cv2.waitKey(0)
        else:
            print("Writing Data")
            extractdatasetImage(I1 , id= id , picNum= picNum)

        picNum += 1
    return

def build_model(inputs):
    x = inputs

    model = Sequential()
    x = Conv2D(input_shape=(img_rows, img_cols, 1), filters=20, kernel_size=(5, 5), padding="same", activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation='relu')(x)
    MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x =  Flatten()(x)
    x = Dense(600, activation="relu")(x)
    output = Dense(42 , activation="softmax")(x)

    model = model(inputs,output,name="LeNet")

    model.summary()

    return model

def predicting(image,isNumber,isChar):
    model = Sequential()
    model.add(
        Conv2D(input_shape=(img_rows, img_cols, 1), filters=20, kernel_size=(5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    sigma = 3
    Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGray2 = cv2.GaussianBlur(Gray, (sigma, sigma), 0)
    th2 = cv2.adaptiveThreshold(imageGray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)
    img_bin = 255 - th2
    kernel = np.ones((2, 2), np.uint8)
    image2 = cv2.dilate(img_bin, kernel, iterations=1)
    image1 = cv2.erode(image2, kernel, iterations=2)
    G1 = imageResize(image1, img_rows, img_cols)
    G1 = img_to_array(G1) / 255
    G1 = np.reshape(G1, (1, img_rows, img_cols, 1))


    # model = build_model(G1)

    if isNumber == True:
        model.add(Dense(11))
        model.add(Activation('softmax'))
        model.load_weights("modelSaves/number.h5")
        predicts = model.predict(G1)
        y_pred = np.argmax(predicts)
        y_pred = classes_adad[y_pred]
    elif isChar == True:
        model.add(Dense(33 , activation="softmax"))
        model.load_weights("modelSaves/char.h5")
        predicts = model.predict(G1)
        y_pred = np.argmax(predicts)
        y_pred = classes_horof[y_pred]
    else:
        # model.add(Dense(800))
        # model.add(Activation('relu'))
        # model.add(Dense(32, activation="softmax"))
        # model.load_weights("Copy of model.h5")
        predicts = model.predict(G1)
        y_pred = np.argmax(predicts)
        y_pred = classes[y_pred]


    return y_pred

def findGraduation(images):
    I = cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY)
    ret, T = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)
    if np.median(T) < 10 or np.mean(T):
        return "کارشناسی"

    I = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    ret, T = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)
    if np.median(T) < 10 or np.mean(T):
        return "کارشناسی ارشد"

    I = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    ret, T = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)
    if np.median(T) < 10 or np.mean(T):
        return "دکتری"

    return "ترک تحصیل!"

def extractFormImage(transformedImage):
    #############extract bytes
    number = []
    numberStr = "number:"
    for k in range(8):
        Xl = 75 + k*68
        Xr = 140 + k*68
        image = transformedImage[435:495, Xl:Xr]
        number.append(image)
        string = "number"
        string += str(k)
        numberStr += predicting(image,True,False)
        G1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        G1 = imageResize(G1, img_rows, img_cols)
        # cv2.imshow(string, G1)




    name = []
    nameStr = "name:"
    for k in range(8):
        Xl = 75 + k * 68
        Xr = 140 + k * 68
        image = transformedImage[530:590, Xl:Xr]
        name.append(image)
        string = "number"
        string += str(k)
        nameStr += predicting(image,False,True)
        cv2.imshow(string, image)

    familyName = []
    familyNameStr = "familyName:"
    for k in range(8):
        Xl = 65 + k * 70
        Xr = 140 + k * 70
        image = transformedImage[610:710, Xl:Xr]
        familyName.append(image)
        string = "number"
        string += str(k)
        familyNameStr += predicting(image,False,True)
        #cv2.imshow(string, image)

    graduation = []
    graduationStr = "graduation:"
    image = transformedImage[720:770, 85:140]
    graduation.append(image)
    image = transformedImage[720:770, 250:300]
    graduation.append(image)
    image = transformedImage[720:770, 470:530]
    graduation.append(image)

    strr = findGraduation(graduation)
    graduationStr += strr

    print(numberStr)
    nameStr = nameStr[::-1]
    print(nameStr)
    print(familyNameStr)
    print(graduationStr)

    # for k in range(3):
    #     string = "number"
    #     string += str(k)
    #     cv2.imshow(string, graduation[k])

def saveImage(image,id,picNum,dataNum,rowNum):
    picName = "/pic" + str(picNum) + "_" + str(dataNum) + ".png"
    # print(picName)

    if (picNum > 228):
        path = "data/test/"
    else:
        path = "data/train/"

    if id [0] == '1':
        if rowNum == 0:
            path += "0"
        elif rowNum == 1:
            path += "1"
        elif rowNum == 2:
            path += "ا"
        elif rowNum == 3:
            path += "ب"
        elif rowNum == 4:
            path += "پ"
        elif rowNum == 5:
            path += "ت"
        elif rowNum == 6:
            path += "ث"
        elif rowNum == 7:
            path += "ج"
        elif rowNum == 8:
            path += "چ"
        elif rowNum == 9:
            path += "ح"
        elif rowNum == 10:
            path += "خ"
        elif rowNum == 11:
            path += "د"
        elif rowNum == 12:
            path += "ذ"
        elif rowNum == 13:
            path += "ر"
        elif rowNum == 14:
            path += "ز"
        elif rowNum == 15:
            path += "ژ"
        elif rowNum == 16:
            path += "س"
        elif rowNum == 17:
            path += "ش"
        elif rowNum == 18:
            path += "ص"
        elif rowNum == 19:
            path += "2"
        elif rowNum == 20:
            path += "3"

    elif id [0] == "2":
        if rowNum == 0:
            path += "4"
        elif rowNum == 1:
            path += "5"
        elif rowNum == 2:
            path += "ض"
        elif rowNum == 3:
            path += "ط"
        elif rowNum == 4:
            path += "ظ"
        elif rowNum == 5:
            path += "ع"
        elif rowNum == 6:
            path += "غ"
        elif rowNum == 7:
            path += "ف"
        elif rowNum == 8:
            path += "ق"
        elif rowNum == 9:
            path += "ک"
        elif rowNum == 10:
            path += "گ"
        elif rowNum == 11:
            path += "ل"
        elif rowNum == 12:
            path += "م"
        elif rowNum == 13:
            path += "ن"
        elif rowNum == 14:
            path += "و"
        elif rowNum == 15:
            path += "ه"
        elif rowNum == 16:
            path += "ی"
        elif rowNum == 17:
            path += "6"
        elif rowNum == 18:
            path += "7"
        elif rowNum == 19:
            path += "8"
        elif rowNum == 20:
            path += "9"
    # print(path)
    grayIm =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path + picName, imageResize(grayIm,img_rows,img_cols))
    return

def extractdatasetImage(transformedImage , id , picNum):
    #############extract bytes
    for k in range(10):
        Xl = 180 + k*81
        Xr = 255 + k*81
        image = transformedImage[10:75, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 0)
        # cv2.imshow(string, image)

    for k in range(10):
        Xl = 180 + k*81
        Xr = 255 + k*81
        image = transformedImage[83:150, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 1)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 95 + k*81
        image = transformedImage[150:245, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 2)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[230:320, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 3)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[325:390, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 4)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[405:470, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 5)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[475:550, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 6)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[555:625, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 7)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[630:710, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 8)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[715:790, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 9)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 15 + k*81
        Xr = 90 + k*81
        image = transformedImage[790:870, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 10)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[870:950, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 11)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[935:1025, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 12)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1015:1105, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 13)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1090:1180, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 14)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1165:1255, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 15)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1240:1330, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 16)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1325:1415, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 17)
        # cv2.imshow(string, image)

    for k in range(14):
        Xl = 10 + k*81
        Xr = 90 + k*81
        image = transformedImage[1405:1495, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 18)
        # cv2.imshow(string, image)

    for k in range(10):
        Xl = 180 + k*81
        Xr = 255 + k*81
        image = transformedImage[1485:1575, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 19)
        # cv2.imshow(string, image)

    for k in range(10):
        Xl = 180 + k * 81
        Xr = 255 + k * 81
        image = transformedImage[1560:1650, Xl:Xr]
        string = ""
        string += str(k)
        saveImage(image, id, picNum, string , 20)
        # cv2.imshow(string, image)


    return
