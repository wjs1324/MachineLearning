############################
# 필요한 라이브러리 import #
############################
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import re

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
import keras.applications.xception as xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

# GPU메모리에 최소한으로 할당 후 필요 프로세스에 따라 메모리 증가를 허락
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#################
# 데이터셋 준비 #
#################
base_path = "./garbage_classification/"

# 12개의 클래스를 dictionary 변수로 저장
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

filenames_list = []
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])
        # listdir(path) : 해당 경로에 있는 모든 파일과 디렉토리 리스트를 반환
    # 이미지 파일명(ex battery1.jpg) 리스트
    filenames_list = filenames_list + filenames
    # 클래스번호(0~12) 리스트
    categories_list = categories_list + [category] * len(filenames)

# 만들어놓은 파일명과 클래스번호 변수를 이용하여 데이터프레임 만들기
df = pd.DataFrame({'filename': filenames_list, 'category': categories_list})
df

# 'filename' 값에 접두사로 클래스명을 추가 (예를 들어, "paper104.jpg" -> "paper/paper104.jpg")
df['filename'] = df['filename'].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
df

# 데이터프레임의 행을 무작위로 섞기
df = df.sample(frac=1).reset_index(drop=True)
    # sample(frac=1) : 전체 행의 100%를 랜덤하게 표본 추출
    # reset_index() : 행 순서가 바껴도 고유인덱스는 안 바뀌기 때문에 인덱스를 다시 재배열(값이 바뀌는 게 아닌 새로운 인덱스 컬럼 생성)
        # drop=True : 기존 인덱스를 삭제할건지 안할건지
df

print('이미지 총 개수는' , len(df))

# 샘플 이미지 참조
random_row = random.randint(0, len(df)-1)
sample = df.iloc[random_row, 0]
print(sample)
sampleimage = image.load_img(base_path + sample)
%matplotlib inline
plt.imshow(sampleimage)


#######################
# 카테고리 분포 시각화 #
#######################
df_visualization = df.copy()
df_visualization['category'] = df_visualization['category'].apply(lambda x:categories[x])
df_visualization
# 전체 드래그
df_visualization['category'].value_counts().plot.bar(x = 'count', y = 'category')
plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class", y=1.02)


##########################
# 학습 신경망 모델 만들기 #
##########################
# 이미지 크기 변수
image_width = 320
image_height = 320
image_size = (image_width, image_height)
image_channels = 3

# 전이 학습?
xception_layer = xception.Xception(include_top = False, input_shape = (image_width, image_height, image_channels),
                       weights = './xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

# We don't want to train the imported weights
xception_layer.trainable = False

model = Sequential()
model.add(Input(shape=(image_width, image_height, image_channels)))

#create a custom layer to apply the preprocessing
def xception_preprocessing(img):
  return xception.preprocess_input(img)

model.add(Lambda(xception_preprocessing))

model.add(xception_layer)
model.add(GlobalAveragePooling2D())
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()


early_stop = EarlyStopping(patience = 2, verbose = 1, monitor='val_categorical_accuracy' , mode='max', min_delta=0.001, restore_best_weights = True)

callbacks = [early_stop]

print('call back defined!')


######################
# Split the Data Set #
######################

#Change the categories from numbers to names
df["category"] = df["category"].replace(categories)

# We first split the data into two sets and then split the validate_df to two sets
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_validate , 'validate size = ', total_validate, 'test size = ', test_df.shape[0])


###################
# Train the model #
###################

batch_size=16

train_datagen = image.ImageDataGenerator(

    ###  Augmentation Start  ###

    #rotation_range=30,
    #shear_range=0.1,
    #zoom_range=0.3,
    #horizontal_flip=True,
    #vertical_flip = True,
    #width_shift_range=0.2,
    #height_shift_range=0.2

    ##  Augmentation End  ###
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size
)


validation_datagen = image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size
)


EPOCHS = 20
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


model.save_weights("model.h5")


##################################
# Visualize the training process #
##################################


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()

ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
ax2.legend()

legend = plt.legend(loc='best')
plt.tight_layout()
plt.show()


#####################
# Evaluate the test #
#####################

test_datagen = image.ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=base_path,
    x_col='filename',
    y_col='category',
    target_size=image_size,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False
)


filenames = test_generator.filenames
nb_samples = len(filenames)


_, accuracy = model.evaluate_generator(test_generator, nb_samples)

print('accuracy on test set = ',  round((accuracy * 100),2 ), '% ')


# We defined at the beginning of this notebook a dictionary that maps the categories number to names, but the train generator
# generated it's own dictionary and it has assigned different numbers to our categories and the predictions made by the model
# will be made using the genrator's dictionary.

gen_label_map = test_generator.class_indices
gen_label_map = dict((v,k) for k,v in gen_label_map.items())
print(gen_label_map)


# get the model's predictions for the test set
preds = model.predict(test_generator, nb_samples)

# Get the category with the highest predicted probability, the prediction is only the category's number and not name
preds = preds.argmax(1)

# Convert the predicted category's number to name
preds = [gen_label_map[item] for item in preds]

# Convert the pandas dataframe to a numpy matrix
labels = test_df['category'].to_numpy()

print(classification_report(labels, preds))


test1 = Image.open("paper/paper16.jpg")
test2 = Image.open("paper/paper2.jpg")
test3 = Image.open("paper/paper3.jpg")
test4 = Image.open("paper/paper10.jpg")
test5 = Image.open("paper/paper5.jpg")

import numpy as np
np.set_printoptions(precision=3)

model(np.expand_dims(np.array(test1.resize((320,320))), axis=0))
model(np.expand_dims(np.array(test2.resize((320,320))), axis=0))
model(np.expand_dims(np.array(test3.resize((320,320))), axis=0))
model(np.expand_dims(np.array(test4.resize((320,320))), axis=0))
model(np.expand_dims(np.array(test5.resize((320,320))), axis=0))
