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


def load_submodel(model_dir, lmodel):
    cnt = 0
    temp_model = tf.keras.models.load_model(model_dir)
    for i, layer in enumerate(temp_model.layers):
        try:
            lmodel.get_layer(layer.name).set_weights(layer.get_weights())
            print('Layer:', layer.name)
            cnt += 1
        except:
            continue
    print(f"{cnt} layer weights loaded")


class SaveLog(tf.keras.callbacks.Callback):
    def __init__(self, save_path='./log/', 
                 save_on_epoch=-1, monitor='loss', mode='auto', 
                 load_prev=False, patience=5, save_best_model=True,
                 restore_best_weights=True,
                 verbose=True):
        
        self.os  = __import__('os')
        self.pkl = __import__('pickle')
        self.np = __import__('numpy')
        self.tf = __import__('tensorflow')
        self.tf.get_logger().setLevel('ERROR')
        super(SaveLog, self).__init__() 
        
        self.save_on_epoch = save_on_epoch
        if save_on_epoch < -1:
            raise ValueError("Invalid 'save_on_epoch' value." 
                             " Should be greater than -1.")

        self.monitor = monitor
        self.save_dir = None
        self.log_dir = None
        self.save_best_model = save_best_model
        self.patience = patience
        self.patience_kept = 0
        self.logs = {}
        self.load_prev = load_prev
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.save_path = save_path
        
        if mode == 'min':
            self.monitor_op = self.np.less
            self.metric_val = self.np.inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = self.np.greater
            self.metric_val = -self.np.inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor:
                self.monitor_op = self.np.greater
                self.metric_val = -self.np.inf
                self.mode = 'max'
            else:
                self.monitor_op = self.np.less
                self.metric_val = self.np.inf
                self.mode = 'min'


    def _save_model(self, best=False):
        sdir = self.best_dir if best else self.save_dir
        if best:
            self.model.save_weights(sdir)
        else:
            self.model.save(sdir, overwrite=True)
        #if self.verbose: 
        print(f'{"Best m" if best else "M"}odel saved on {sdir}')  


    def _load_model(self):
        try:
            self.model.load_weights(self.save_dir)
            if self.verbose: print('Weights loaded')
            return
        except:
            pass
        try:
            cnt = 0
            temp_model = self.tf.keras.models.load_model(self.save_dir)
            for i, layer in enumerate(temp_model.layers):
                try:
                    self.model.get_layer(layer.name).set_weights(layer.get_weights())
                    cnt += 1
                except:
                    continue
            if self.verbose: print(f"{cnt} layer weights loaded")
        except:
            pass


    def _save_logs(self, epoch):
        with open(self.log_dir, 'wb') as f:
            self.pkl.dump(self.logs, f)
        if self.verbose: print(f'Logs saved on epoch {epoch+1}')


    def _load_logs(self):
        try:
            with open(self.log_dir, 'rb') as f:
                log = self.pkl.load(f)
            if self.mode == 'max':
                self.metric_val = max(log[self.monitor])
            else:
                self.metric_val = min(log[self.monitor])
            if self.verbose: print(f'Logs loaded, best {self.monitor}: {self.metric_val:.5f}')
            return log
        except:
            return {}


    def _saver(self, epoch, logs):
        log_val = logs[self.monitor]
        if self.save_on_epoch == 0:
            self._save_logs(epoch)
            self._save_model()

            if self.save_best_model and self.monitor_op(log_val, self.metric_val):
                self.metric_val = log_val
                self._save_model(best=True)

        elif self.save_on_epoch == -1:
            if self.monitor_op(log_val, self.metric_val):
                self.metric_val = log_val
                if self.verbose: print('Minimum loss found')
                self._save_logs(epoch)
                if self.save_best_model:
                    self._save_model(best=True)
        else:
            if epoch % self.save_on_epoch == 0:
                self._save_logs(epoch)
                self._save_model()
            
            if self.save_best_model and self.monitor_op(log_val, self.metric_val):
                self.metric_val = log_val
                self._save_model(best=True)
                

    def on_train_begin(self, logs=None):
        self.save_dir = self.os.path.join(self.save_path, self.model.name, '')
        self.best_dir = self.os.path.join(self.save_dir, 'best_weight.h5')
        self.log_dir = self.os.path.join(self.save_dir, 'train_log.pkl')
         
        if self.verbose: print('SavePath', self.save_dir)
        if not self.os.path.exists(self.save_dir):
            self.os.makedirs(self.save_dir)
            if self.verbose: print('New directory created')
        elif self.load_prev:
            self._load_model()
            self.logs = self._load_logs() or {}


    def on_epoch_end(self, epoch, logs=None):
        for key, val in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(val)
        
        if self.patience != None:
            if self.monitor_op(logs[self.monitor], self.metric_val):
                self.patience_kept = 0
            else:
                self.patience_kept += 1
            if self.patience_kept > self.patience:
                self.model.stop_training = True
                if self.verbose: print("Stopping training")
                if self.restore_best_weights:
                    if self.verbose: print("Restoring best weights")
                    self.model.load_weights(self.best_dir)
        
        self._saver(epoch, logs)

epochs = 1000
early_stopping_patience = 10


dirs = ['C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\new_weight\\ocr_DenseNet121_GRU', 
        'C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\new_weight\\ocr_NASNetMobile_GRU',
        'C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\BHWR\\new_weight\\ocr_NASNetMobile_LSTM',]

def get_flops(model_h5_path):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


for d in dirs[1:]:
    print(d.split('/')[-1], ":", get_flops(d))


# Train the model
for baseline in list(layer_name.keys())[3:]:
    model = build_model(baseline, "GRU")
 
    logger = SaveLog(save_path='C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\new_weight', 
                    monitor='val_loss', 
                    save_on_epoch=5, patience=early_stopping_patience, 
                    verbose=False, load_prev=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        verbose=0,
        callbacks=[logger],
        max_queue_size=20,
    )
 
    #========================================
    model = build_model(baseline, "LSTM")
 
    logger = SaveLog(save_path='C:\\Users\\subha\\OneDrive\\Desktop\\Word\\Two\\BHWR\\new_weight', 
                    monitor='val_loss', 
                    save_on_epoch=5, patience=early_stopping_patience, 
                    verbose=False, load_prev=True)
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        verbose=0,
        callbacks=[logger],
        max_queue_size=20,
    )
