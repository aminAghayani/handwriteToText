import os
from dataExtractor import extract
import cv2

pathData = "data"
pathDataTrain = pathData + "/train"
pathDataTest = pathData + "/test"
pathSampleForms = 'sampleForms/form.png'
pathWrittenForms = 'writtenForms'
pathSampleFormsDataset = 'sampleForms/dataset.png'
pathRawData = 'rawData'

dir = os.listdir(pathData)
if len(dir) == 0:
    print("Processing raw Data...")
    os.mkdir(pathDataTrain)
    os.mkdir(pathDataTest)
    pathTrain = "data/train"
    os.mkdir(pathDataTrain + "/0")
    os.mkdir(pathDataTrain + "/1")
    os.mkdir(pathDataTrain + "/2")
    os.mkdir(pathDataTrain + "/3")
    os.mkdir(pathDataTrain + "/4")
    os.mkdir(pathDataTrain + "/5")
    os.mkdir(pathDataTrain + "/6")
    os.mkdir(pathDataTrain + "/7")
    os.mkdir(pathDataTrain + "/8")
    os.mkdir(pathDataTrain + "/9")
    os.mkdir(pathDataTrain + "/ا")
    os.mkdir(pathDataTrain + "/ب")
    os.mkdir(pathDataTrain + "/پ")
    os.mkdir(pathDataTrain + "/ت")
    os.mkdir(pathDataTrain + "/ث")
    os.mkdir(pathDataTrain + "/ج")
    os.mkdir(pathDataTrain + "/چ")
    os.mkdir(pathDataTrain + "/ح")
    os.mkdir(pathDataTrain + "/خ")
    os.mkdir(pathDataTrain + "/د")
    os.mkdir(pathDataTrain + "/ذ")
    os.mkdir(pathDataTrain + "/ر")
    os.mkdir(pathDataTrain + "/ز")
    os.mkdir(pathDataTrain + "/ژ")
    os.mkdir(pathDataTrain + "/س")
    os.mkdir(pathDataTrain + "/ش")
    os.mkdir(pathDataTrain + "/ص")
    os.mkdir(pathDataTrain + "/ض")
    os.mkdir(pathDataTrain + "/ط")
    os.mkdir(pathDataTrain + "/ظ")
    os.mkdir(pathDataTrain + "/ع")
    os.mkdir(pathDataTrain + "/غ")
    os.mkdir(pathDataTrain + "/ف")
    os.mkdir(pathDataTrain + "/ق")
    os.mkdir(pathDataTrain + "/ک")
    os.mkdir(pathDataTrain + "/گ")
    os.mkdir(pathDataTrain + "/ل")
    os.mkdir(pathDataTrain + "/م")
    os.mkdir(pathDataTrain + "/ن")
    os.mkdir(pathDataTrain + "/و")
    os.mkdir(pathDataTrain + "/ه")
    os.mkdir(pathDataTrain + "/ی")

    path = "data/test"
    os.mkdir(pathDataTest + "/0")
    os.mkdir(pathDataTest + "/1")
    os.mkdir(pathDataTest + "/2")
    os.mkdir(pathDataTest + "/3")
    os.mkdir(pathDataTest + "/4")
    os.mkdir(pathDataTest + "/5")
    os.mkdir(pathDataTest + "/6")
    os.mkdir(pathDataTest + "/7")
    os.mkdir(pathDataTest + "/8")
    os.mkdir(pathDataTest + "/9")
    os.mkdir(pathDataTest + "/ا")
    os.mkdir(pathDataTest + "/ب")
    os.mkdir(pathDataTest + "/پ")
    os.mkdir(pathDataTest + "/ت")
    os.mkdir(pathDataTest + "/ث")
    os.mkdir(pathDataTest + "/ج")
    os.mkdir(pathDataTest + "/چ")
    os.mkdir(pathDataTest + "/ح")
    os.mkdir(pathDataTest + "/خ")
    os.mkdir(pathDataTest + "/د")
    os.mkdir(pathDataTest + "/ذ")
    os.mkdir(pathDataTest + "/ر")
    os.mkdir(pathDataTest + "/ز")
    os.mkdir(pathDataTest + "/ژ")
    os.mkdir(pathDataTest + "/س")
    os.mkdir(pathDataTest + "/ش")
    os.mkdir(pathDataTest + "/ص")
    os.mkdir(pathDataTest + "/ض")
    os.mkdir(pathDataTest + "/ط")
    os.mkdir(pathDataTest + "/ظ")
    os.mkdir(pathDataTest + "/ع")
    os.mkdir(pathDataTest + "/غ")
    os.mkdir(pathDataTest + "/ف")
    os.mkdir(pathDataTest + "/ق")
    os.mkdir(pathDataTest + "/ک")
    os.mkdir(pathDataTest + "/گ")
    os.mkdir(pathDataTest + "/ل")
    os.mkdir(pathDataTest + "/م")
    os.mkdir(pathDataTest + "/ن")
    os.mkdir(pathDataTest + "/و")
    os.mkdir(pathDataTest + "/ه")
    os.mkdir(pathDataTest + "/ی")
# extract(pathSampleFormsDataset, "aslia", isForm=False)

# extract(pathSampleForms, pathWrittenForms , isForm=True)


