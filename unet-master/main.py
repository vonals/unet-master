from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.config.experimental_run_functions_eagerly(True)

#数据增强参数
data_gen_args = dict(rotation_range=0.2,  # 旋转
                    width_shift_range=0.05,  # 宽度变化
                    height_shift_range=0.05,  # 高度变化
                    shear_range=0.05,  #
                    zoom_range=0.05,  # 缩放
                    horizontal_flip=True,  # 水平旋转
                    fill_mode='nearest')
# 加载数据集
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir = None)
# 加载网络模型
model = unet()
# callback
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
tb_callback = TensorBoard(log_dir="./logs", histogram_freq=1, embeddings_freq=1)
# 训练网络
model.fit_generator(myGene,steps_per_epoch=50, epochs=1, callbacks=[tb_callback])
# 测试网络
testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
# results = model.evaluate(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)
