import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
import nltk
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
# os.system('pip install -U git+https://github.com/albumentations-team/albumentations')
# os.system('pip install -U git+https://github.com/tensorflow/tensorflow')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import Lambda

from keras.preprocessing.image import ImageDataGenerator
import albumentations as albu
from tensorflow.keras.applications import *

import pydot

input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
x = tf.keras.layers.Embedding(
    output_dim=512, input_dim=10000, input_length=100)(input)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
model = tf.keras.Model(inputs=[input], outputs=[output])
dot_img_file = 'IMG_20240104_223208.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


data_dir = 'C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\wordlevel_data'

data_dir = r'C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\wordlevel_data'
dirs = os.listdir(data_dir)

def resize():
    for item in dirs:
        if os.path.isfile(data_dir+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,50), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=100)            
resize()

def filter(imgs, lbls):
    ret_imgs = []
    ret_lbls = []
 
    for i, lbl in enumerate(lbls):
        with open(lbl, 'r', encoding='utf-8') as f:
            txt = f.read()
            ignore = False
            
            if len(txt) > 10: 
                ignore = True
                continue
            
            for c in txt:
                id = ord(c)
                ignore = ignore or (ord('a') <= id and id <= ord('z'))
                ignore = ignore or (ord('A') <= id and id <= ord('Z'))
                ignore = ignore or (ord('0') <= id and id <= ord('9'))
                ignore = ignore or (c in ['%', "'", '(', ')', '*', ',', '-', '.', '/', '\\', 'θ', 'π',])
            
            if not ignore:
                ret_imgs.append(imgs[i])
                ret_lbls.append(lbls[i])
 
    return ret_imgs, ret_lbls


data_dir_path = Path(data_dir)
 
images = sorted(list(map(str, list(data_dir_path.glob("*.jpg")))))
labels = sorted(list(map(str, list(data_dir_path.glob("*.txt")))))
images, labels = filter(images, labels)
 
img_width = 200
img_height = 50
 
print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
 
texts = []

for label in labels:
    f = open (label, 'r', encoding='utf-8')
    text = f.read()
    texts.append(text)
    f.close()
 
characters = set(char for txt in texts for char in txt)
print("Characters present: ", characters)
print("Number of unique characters: ", len(characters))
 
max_length = max([len(txt) for txt in texts])
 
print('Max length', max_length)
characters = sorted(list(characters))
print(characters)


# print(labels[:10])

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=characters, num_oov_indices=0, #mask_token=" "
)
 
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), num_oov_indices=1, invert=True,
    oov_token=""
)
 
x_train, x_valid, y_train, y_valid = train_test_split(np.array(images), 
                                                      np.array(texts),
                                                      random_state=42,
                                                      test_size=0.30)

x_valid, x_test, y_valid, y_test = train_test_split(x_valid, 
                                                    y_valid,
                                                    random_state=42,
                                                    test_size=0.50)


# print(x_train[:10])
# print(y_train[:10])
# print(len(x_train))
# print(len(x_valid))
# print(len(x_test))



def strong_aug(img_shape, p=0.6, color_augment=False):
    # horizontal
    hh = int(img_shape[0]*0.8)
    hw = int(img_shape[1]*0.08)
    vh = int(img_shape[0]*0.04)
    vw = int(img_shape[1]*0.8)
    return albu.Compose([
        albu.OneOf([
            albu.Cutout(num_holes=4, max_h_size=vh, max_w_size=vw),
            albu.Cutout(num_holes=4, max_h_size=hh, max_w_size=hw),
        ], p=0.3),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=0.6),

        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, 
                              rotate_limit=20, p=0.4),

        albu.OneOf([
            albu.OpticalDistortion(p=0.4),
            albu.GridDistortion(p=0.4),
            albu.IAAPiecewiseAffine(p=0.2),
        ], p=0.5),
    ], p=p)



def adjust_unicode(character, offset):
    unicode_value = ord(character)
    adjusted_value = unicode_value + offset
    adjusted_character = chr(adjusted_value)
    return adjusted_character

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, X_dirs, y_labels, augment, batch_size=16,
                 output_shape = (50, 200, 3), p=0.6):
        
        self.img_height     = output_shape[0]
        self.img_width      = output_shape[1]
        self.channels       = output_shape[2]
        self.augment        = augment
        self.batch_size     = batch_size
        self.indexes        = np.arange(y_labels.shape[0])
 
        self.augmentation   = strong_aug((self.img_width, self.img_height, 
                                          self.channels), p=p)
 
        self.x_dirs         = X_dirs
        self.y_labels       = y_labels
 
        self.on_epoch_end()

    def __len__(self):
        return int(self.y_labels.shape[0] // self.batch_size)
 
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
 
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imgs = tf.map_fn(lambda id: tf.transpose(tf.image.resize(tf.io.decode_png(tf.io.read_file(self.x_dirs[id]), 
                                                     channels=1), 
                                                    [self.img_height, self.img_width]), perm=[1, 0, 2]), 
                         indexes, dtype=tf.float32)
        
        imgs = keras.backend.repeat_elements(x=imgs, rep=self.channels, axis=-1)
        imgs = (imgs.numpy()).astype(np.uint8)
        
        labels = char_to_num(tf.strings.unicode_split(self.y_labels[indexes], input_encoding="UTF-8")).to_tensor()
 
        if self.augment:
            return {"image": tf.keras.applications.imagenet_utils.preprocess_input(self.aug(imgs)), "txt": labels}
        return {"image": tf.keras.applications.imagenet_utils.preprocess_input(imgs), "txt": labels}
 
    def read_file(self, fileID):
        img = tf.io.read_file(self.x_dirs[fileID])
        img = tf.io.decode_png(img, channels=1)
        return img
 
    def aug(self, imgs):
        for i, image in enumerate(imgs):
           imgs[i] = self.augmentation(image=image)['image']
 
        return imgs
    


# text = "আইঈ"
# encoded_text = text.encode('utf-8')
# print(encoded_text)
# decoded_text = encoded_text.decode('utf-8')
# print(decoded_text)



test_models = ['VGG16', 'VGG19', 'Xception', 'ResNet50', 'ResNet101', 
               'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 
               'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'MobileNetV2',
               'DenseNet121', 'DenseNet169', 'DenseNet201']
 
def CustomNetwork(model_name, layer_name=None, 
                  input_shape=(img_width, img_height, 3),
                  add_activation=False):
    
    tf.keras.backend.clear_session()
    inp = layers.Input(input_shape)
    model_str = model_name + "(input_tensor=inp, include_top=False, weights='imagenet')"
    base_model = eval(model_str)
 
    if layer_name == None:
        return base_model
    
    out = base_model.get_layer(layer_name).output
    if add_activation:
        out = layers.ReLU()(out)
 
    return keras.models.Model(inp, out, name=f"{model_name}")
 
 
baseline_name = "DenseNet121"
layer_name = {
    "MobileNet": "conv_pw_5_relu",
    "DenseNet121": "pool3_conv",
    "Xception": "block4_sepconv2",
    "NASNetMobile":"activation_71",
}
 
baseline = CustomNetwork(baseline_name, layer_name[baseline_name], 
                         add_activation=False)
baseline.summary()




train_dataset = DataGen(x_train, y_train, augment=True, p=0.65)
validation_dataset = DataGen(x_valid, y_valid, augment=False)
test_dataset = DataGen(x_test, y_test, augment=False)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
 
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
 
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
 
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
 
        return y_pred
 
def build_model(baseline_name, rnn="LSTM"):
    if rnn not in ["LSTM", "GRU"]:
        raise ValueError("INVALID RNN TYPE")

    input_img = layers.Input(
        shape=(img_width, img_height, 3), name="image", dtype="float32"
    )
 
    base = CustomNetwork(baseline_name, layer_name[baseline_name], 
                         add_activation=False)
    x = base(input_img)
    
    print('pass 3', x.shape)
    _, w, h, c = x.shape
    x = layers.Reshape(target_shape=(w, h*c), name="reshape")(x)
 
    texts = layers.Input(name="txt", shape=(None,), dtype="float32")

    print("Bidirectional layer input shape", x.shape)

    if rnn == "LSTM":
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.1,
                                          name="LSTM1"), name="BD1")(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.1,
                                          name="LSTM2"), name="BD2")(x)
    else:
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.1,
                                          name="GRU1"))(x)
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.1,
                                          name="GRU2"))(x)
 
    x = layers.Dense(len(char_to_num.get_vocabulary())+1, 
                     activation="softmax", name="dense2",
                     use_bias=True)(x)
 
    output = CTCLayer(name="ctc_loss")(texts, x)
 
    model = keras.models.Model(
        inputs=[input_img, texts], outputs=output, name=f"ocr_{base.name}_{rnn}",
        #inputs=base.input, outputs=output, name="ocr_model_v1"
    )
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model
 
model = build_model("DenseNet121", "LSTM")
model.summary()




history = model.fit(
        DataGen(x_train, y_train, True, p=1),
        epochs=1,)