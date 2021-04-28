# 主函数
#
from model import *
from data import *
# from net1 import *
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 调试时使用（影响运行速度）
# tf.config.experimental_run_functions_eagerly(True)

# 数据增强参数
# 训练集参数
data_gen_args = dict(rotation_range=0.2,  # 旋转
                     width_shift_range=0.05,  # 宽度变化
                     height_shift_range=0.05,  # 高度变化
                     shear_range=0.05,  #
                     zoom_range=0.05,  # 缩放
                     horizontal_flip=True,  # 水平旋转
                     fill_mode='nearest')
# 验证集参数
val_gen_args = dict()

# 加载数据集
# 训练集
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
# 验证集
valGene = trainGenerator(2, 'data/membrane/validation', 'image', 'mask', val_gen_args, save_to_dir=None)
# 测试集
testGene = testGenerator("data/membrane/test")

# 加载网络模型（测试多个）
model = unet()
# model = net1()

# keras 回调函数
# callback EarlyStopping(monitor='acc', patience=1), ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=1)
callbacks_list = [EarlyStopping(monitor='accuracy', patience=2),
                  ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
                  ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True),
                  TensorBoard(log_dir="./logs", histogram_freq=1, embeddings_freq=1)
                  ]
# 训练网络
history = model.fit_generator(myGene, steps_per_epoch=32, epochs=20, callbacks=callbacks_list,
                              validation_data=valGene, validation_steps=2)

results = model.predict_generator(testGene, 30, verbose=1)
# results = model.evaluate(testGene, 30, verbose=1)
# 保存分割后的图像
saveResult("data/membrane/test", results)
# # 显示loss和val_loss（过拟合可视化）
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
