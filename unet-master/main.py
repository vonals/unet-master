# 主函数
#
# from model import *
from NewUNet import *
# from FCN import *
from NestedUNet import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 调试时使用（tf静态图模式，影响运行速度）
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
# model = UNet('unet_membrane.hdf5')
# model = UNet1('unet_membrane.hdf5')
# model = UNet1()
# model = FCN()
model = NestedUNet('unet_membrane.hdf5', using_deep_supervision=True)
# model = NestedUNet(using_deep_supervision=True)
# model = NestedUNet()
# model = NestedUNet('unet_membrane.hdf5')


# keras 回调函数
# callback
# EarlyStopping(monitor='acc', patience=3),
# , ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2)
# callbacks 通常指标 val_loss
# deep supervision使用指标   val_output_4_loss
callbacks_list = [
                  ReduceLROnPlateau(monitor="val_output_4_loss", factor=0.1, patience=2),
                  ModelCheckpoint('unet_membrane.hdf5', monitor='val_output_4_loss', verbose=1, save_best_only=True),
                  TensorBoard(log_dir="./logs", histogram_freq=1, embeddings_freq=1)
                  ]
# 训练网络
# callbacks=callbacks_list,
model.fit_generator(myGene, steps_per_epoch=14, epochs=10, callbacks=callbacks_list,
                    validation_data=valGene, validation_steps=1)

# 得到分割结果
results = model.predict(testGene, 30, verbose=1)
# 保存分割后的图像
# saveResult("data/membrane/test", results, deep_supervision=True)
saveResult("data/membrane/test", results, deep_supervision=True, mode_accuracy=True)
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
