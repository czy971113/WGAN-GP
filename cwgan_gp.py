from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding,Conv1D,MaxPooling1D,GlobalAveragePooling1D,UpSampling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from functools import partial
import keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np
from load_matdata import load_matdata
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

class RandomWeightedAverage(_Merge):

    def _merge_function(self, inputs):
        global batch_size
        alpha = K.random_uniform((batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
class CWGANGP():
    def __init__(self, epochs=100, batch_size=32):
        self.data_shape=(1024,1)
        self.nclasses = 2   
        self.latent_dim = 140
        self.losslog = []
        self.epochs = epochs
        self.batch_size = batch_size
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 3
        self.n_generator = 2
        optimizer_c = RMSprop(lr=0.00001)
        optimizer_g = RMSprop(lr=0.00001)

        # Build the generator and critic
       
        self.generator = self.build_generator()   
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_data = Input(shape=self.data_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,1))
        
        # Generate image based of noise (fake sample) and add label to the input 
        label = Input(shape=(1,))
        fake_data = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_data, label])
        valid = self.critic([real_data, label])

        # Construct weighted average between real and fake images
        interpolated_data = RandomWeightedAverage()([real_data, fake_data])
          
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_data, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_data)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_data, label, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss = [self.wasserstein_loss,
                                          self.wasserstein_loss,
                                          partial_gp_loss],
                                          optimizer = optimizer_c,
                                          loss_weights = [1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,1))
        # add label to the input
        label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([img, label])
        # Defines generator model
        self.generator_model = Model([z_gen, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer_g )
        
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
 
        # (140,1)->(1024,1)
        model = Sequential()
        model.add(Dense(1, input_shape=(self.latent_dim,1), activation='relu'))
        model.add(UpSampling1D(2))
        model.add(Conv1D(filters=128, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=128, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=128, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=128, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(UpSampling1D(2))
        model.add(Conv1D(filters=64, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=64, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=64, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=64, kernel_size=5, strides=1,padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(UpSampling1D(2))
        model.add(Conv1D(filters=16, kernel_size=5, strides=1,padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=16, kernel_size=5, strides=1,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=16, kernel_size=5, strides=1,padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=16, kernel_size=5, strides=1,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=1, kernel_size=5, strides=1,padding='same'))
        model.add(Activation('tanh'))

        #(140,1)+(1,1)->(140,1)
        noise = Input(shape=(self.latent_dim,1))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.nclasses, self.latent_dim)(label))
        label_embedding = Reshape((self.latent_dim,1))(label_embedding)
        model_input = multiply([noise, label_embedding])
        data = model(model_input)
        model = Model([noise, label], data)

        return model

    def build_critic(self):

        model = Sequential()

        model.add(Conv1D(filters=64, kernel_size=5, padding='valid',strides=2,input_shape=(1024,1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=64, kernel_size=5, padding='valid',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=64, kernel_size=5, padding='valid',strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=64, kernel_size=5, padding='valid',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(pool_size=5, strides=2))
        model.add(Conv1D(filters=128, kernel_size=5, padding='valid',strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=128, kernel_size=5, padding='valid',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, kernel_size=5, padding='same',strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=128, kernel_size=5, padding='same',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(pool_size=5, strides=2))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same',strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same',strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same',strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1))

        data = Input(shape=(1024,1))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.nclasses, 1024)(label))
        label_embedding = Reshape((1024,1))(label_embedding)
        flat_data = Flatten()(data)
        flat_data = Reshape((1024,1))(flat_data)
        model_input = multiply([flat_data, label_embedding])        
        validity = model(model_input)
        model = Model([data, label], validity)
        return model    
    

    def train(self):

        # Load the dataset
        data_zc = load_matdata(path='gan_dataset/zc.mat',key='data')
        print(data_zc.shape)
        data_bph = load_matdata(path='gan_dataset/bph.mat',key='bph')
        print(data_bph.shape)

        data_zc_X = data_zc[:,0:1024]
        data_zc_Y = data_zc[:,1024]
        data_bph_X = data_bph[0:300,0:1024]
        data_bph_Y = data_bph[0:300,1024]

        data_X = np.concatenate((data_zc_X,data_bph_X),axis=0)
        data_Y = np.concatenate((data_zc_Y,data_bph_Y),axis=0)
        print(data_X.shape)
        print(data_Y.shape)

        # Rescale -1 to 1

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) 

        for epoch in range(self.epochs):
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, data_X.shape[0], self.batch_size)
                datas, labels = data_X[idx], data_Y[idx]
                datas = datas.reshape(self.batch_size,1024,1)
                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                noise = np.reshape(noise,(self.batch_size,self.latent_dim,1))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([datas, labels, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            for n in range(self.n_generator):
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                noise = np.reshape(noise,(self.batch_size,self.latent_dim,1))
                sampled_labels = np.random.randint(0, self.nclasses, self.batch_size).reshape(-1, 1)
                g_loss = self.generator_model.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            self.losslog.append([d_loss[0], g_loss])
            

if __name__ == '__main__':
    epochs = 20000
    batch_size = 32
    sample_interval = 50
    wgan = CWGANGP(epochs, batch_size)
    wgan.train()