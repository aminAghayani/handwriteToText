import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten



######highper parameters
train_data_path = "data/train"
test_data_path = "data/test"
img_rows = 28
img_cols = 28
batch_size = 32
num_classes = 42
EPOCHS = 10
BS = 64


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical' ,
                                                    color_mode="grayscale" )

test_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = test_datagen.flow_from_directory(test_data_path , target_size=(img_rows , img_cols)
                                                        , color_mode="grayscale" , shuffle = False)


# Build model
model = Sequential()
model.add(Conv2D(input_shape=(img_rows,img_cols,1) ,filters=20, kernel_size=(5, 5), padding="same" ,activation= 'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=50, kernel_size=(5, 5), padding="same" ,activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath="model.h5",
                             monitor="val_acc",
                             verbose=1,
                             save_best_only=True)

model.summary()

# model.load_weights("model.h5")

# #Train
training_log = model.fit_generator(train_generator,epochs=EPOCHS,steps_per_epoch= 1000,callbacks=[checkpoint],
                                   validation_data=validation_generator,validation_steps= 300)


model.save("modelSaves/All.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(EPOCHS), training_log.history["loss"], label="train_loss")
# plt.plot(np.arange(EPOCHS), training_log.history["acc"], label="train_acc")
plt.plot(np.arange(EPOCHS), training_log.history["val_loss"], label="val_loss")
# plt.plot(np.arange(EPOCHS), training_log.history["val_acc"], label="val_acc")
plt.ylabel("loss/accuracy")
plt.title("training plot")
plt.legend(loc="middle right")
plt.savefig("training_plot.png")
plt.show()


#####testing
model.load_weights("modelSaves/All.h5")


Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
conf = confusion_matrix(validation_generator.classes, y_pred)
print(conf)
print('Classification Report')
target_names = ['0','1','2','3','4','5','6','7','8','9' ,
                'ا','ب','پ','ت','ث','ج','چ','ح','خ','د' ,
                'ذ','ر','ز','ژ','س','ش','ص','ض','ط','ظ','ع',
                'غ','ف','ق','ک','گ','ل','م','ن','و','ه' ,
                'ی']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

plt.figure()
conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(conf, range(42), range(42))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
plt.savefig("confusion_matrixTest.png")
plt.show()




train_valid = ImageDataGenerator(rescale=1. / 255)
validation_train = train_valid.flow_from_directory(train_data_path , target_size=(img_rows , img_cols), color_mode="grayscale"
                                                        , shuffle=False)

Y_pred = model.predict_generator(validation_train)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
conf = confusion_matrix(validation_train.classes, y_pred)
print(conf)
print('Classification Report')
target_names = ['0','1','2','3','4','5','6','7','8','9' ,
                'ا','ب','پ','ت','ث','ج','چ','ح','خ','د' ,
                'ذ','ر','ز','ژ','س','ش','ص','ض','ط','ظ','ع',
                'غ','ف','ق','ک','گ','ل','م','ن','و','ه' ,
                'ی']
print(classification_report(validation_train.classes, y_pred, target_names=target_names))

plt.figure()
conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(conf, range(42), range(42))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
plt.savefig("confusion_matrixTrain.png")
plt.show()




