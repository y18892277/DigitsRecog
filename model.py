import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import json

# GPU 检测与内存按需增长设置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置TensorFlow只在需要时申请GPU显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"检测到 {len(gpus)} 个物理GPU, 配置了 {len(logical_gpus)} 个逻辑GPU.")
        print("TensorFlow 将会使用 GPU 进行训练!")
    except RuntimeError as e:
        # 显存增长必须在GPU初始化之前设置
        print(f"GPU内存按需增长设置失败: {e}")
        print("程序仍会尝试使用GPU，但如果遇到显存问题，请注意。")
else:
    print("未检测到兼容的GPU，TensorFlow 将使用 CPU 进行训练。")
    print("如果希望使用GPU，请确保CUDA和cuDNN已正确安装并与TensorFlow版本兼容。")


batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(input_shape)

# 构建网络
model = Sequential()

# 第一个卷积层块
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 第二个卷积层块
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# 全连接层块
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# 输出层
model.add(Dense(10, activation='softmax'))

model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转 ±10 度
    width_shift_range=0.1,  # 随机水平平移 ±10%
    height_shift_range=0.1, # 随机垂直平移 ±10%
    zoom_range=0.1          # 随机缩放 ±10%
)
datagen.fit(x_train)

# 早停回调
early_stopping = EarlyStopping(
    monitor='val_accuracy', # 监控验证集准确率
    patience=5,             # 如果连续5个epoch没有提升则停止
    verbose=1,
    restore_best_weights=True # 恢复在验证集上表现最好的权重
)

# 训练模型
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping], # 加入早停回调
          steps_per_epoch=len(x_train) // batch_size # 因为使用了 datagen.flow
         )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
    
model_file = 'model.h5'
model.save(model_file)
print(f"模型已保存到 {model_file} 和 model.json")