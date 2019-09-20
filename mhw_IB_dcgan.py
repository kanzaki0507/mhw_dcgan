import os
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Activation, GlobalAveragePooling2D, Input, BatchNormalization, Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

class DCGAN():
    def __init__(self, img_size=(64,64), img_channels = 3, ans_path = "./pokemon64/", save_path = "./newMonster", save_name = "pokemon" ):
        # 画像データ用の入力データサイズ
        self.img_rows, self.img_cols = img_size
        self.channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        #画像の保存用のパス
        self.save_path = save_path
        self.save_name = save_name
        os.makedirs(save_path,exist_ok=True)

        #教師データのセット
        self.ans_path = ans_path
        self.names = os.listdir(ans_path)
        self.names.sort()
        self.X_train = []
        i = 1
        for name in self.names:
            img = cv2.imread(ans_path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.X_train.append(img)
        self.X_train = (np.array(self.X_train) / 127.5) - 1.0


        # 変数類のセット
        self.z_dim = 100
        optimizer = Adam(0.0002, 0.5)

        # discriminator学習用コンパイル
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Generator学習用コンパイル
        self.generator = self.Generator()
        self.combined = self.Combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def Generator(self):
        noise_shape = (self.z_dim,)
        model = Sequential()
        model.add(Dense(128 * 16 * 16, activation="relu", input_shape=noise_shape))
        model.add(Reshape((16, 16, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        return model

    def Discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(2, activation='sigmoid'))
        model.summary()

        return model

    def Combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def save_imgs(self, epoch):
        # 生成画像を敷き詰めるときの行数、列数
        r, c = 7, 10
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        # 生成画像を0-1に再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.save_path + self.save_name + "%d.png" % epoch)

    def train(self, epochs=100000, batch_size=32, itv=100):
        for epoch in range(epochs):
            imgs_true = np.array(self.X_train)[np.random.randint(0, len(self.X_train)-1 , batch_size//2)]
            noise = np.random.normal(0, 1, (batch_size//2, self.z_dim))
            imgs_fake = self.generator.predict(noise)
            batch_true = np.zeros([batch_size, 2])
            batch_true[:,0] = 1
            batch_false = np.zeros([batch_size, 2])
            batch_false[:,1] = 1
            # discriminatorの学習
            d_loss_real = self.discriminator.train_on_batch(imgs_true, batch_true[:batch_size//2])
            d_loss_fake = self.discriminator.train_on_batch(imgs_fake, batch_false[:batch_size//2])
            # generatorの学習
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, batch_true)

            print("epoch=%d:  (discriminator's learning) d_loss_real: %f, d_loss_fake: %f (generater's_learning)leaning loss: %f" % (epoch, d_loss_real[0], d_loss_fake[0], g_loss))
            if epoch % itv == 0:
                self.save_imgs(epoch)



#実装部分
if __name__ == '__main__':
    gan = DCGAN(ans_path = "./img/", save_path = "./newMonster/", save_name = "monster")
    gan.Generator()
    gan.Discriminator()
    gan.Combined()
    gan.train(epochs = 10000, itv=1000)