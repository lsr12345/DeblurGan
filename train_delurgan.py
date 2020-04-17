
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
# from keras.layers.merge import _Merge

import tensorflow_addons as tfa

import numpy as np
import cv2
import datetime
import time
import os

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[2]:


from switchnorm import SwitchNormalization
keras.utils.get_custom_objects().update({'SwitchNormalization': SwitchNormalization})


# In[3]:


lr = 1e-4
batch_size = 1
LAMBDA = 1.0  # 100
gradient_penalty_weight = 10.0

input_shape = (256, 256, 3)  # (360, 640, 3)
ngf = 64  #定义生成器原始卷积核个数
ndf = 64#定义判别器原始卷积核个数
input_nc = 3#定义输入通道
output_nc = 3#定义输出通道

n_blocks_gen = 9#定义残差层数量


# In[4]:


log_dir="./logs/"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# if not os.path.exists(checkpoint_prefix):
#     os.mkdir(checkpoint_prefix)


# In[5]:


def generate_images(model, test_input, epoch):
    test_input = test_input.astype(np.float64)
    test_input = np.expand_dims(test_input, axis=0)
    prediction = model(test_input, training=False)
    prediction = prediction.numpy()
    cv2.imwrite(visual_dir+'epoch:{}_visualization.jpg'.format(epoch), prediction[0])
    
# generate_images(generator, input_test, 1)

def load_img(path_A, path_B):
    blur_image = tf.io.read_file(path_A)
    blur_image = tf.io.decode_jpeg(blur_image)

    sharp_image = tf.io.read_file(path_B)
    sharp_image = tf.io.decode_jpeg(sharp_image)

    
    blur_image = tf.cast(blur_image, tf.float32)
    sharp_image = tf.cast(sharp_image, tf.float32)
    
    return blur_image, sharp_image

def resize_img(blur_image, sharp_image, input_shape):
    blur_image = tf.image.resize(blur_image, [input_shape[0], input_shape[1]],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    sharp_image = tf.image.resize(sharp_image, [input_shape[0], input_shape[1]],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return blur_image, sharp_image

def random_crop(blur_image, sharp_image, input_shape):
    stacked_image = tf.stack([blur_image, sharp_image], axis=0)
    croped_image = tf.image.random_crop(stacked_image,
                                        size=[2, input_shape[0], input_shape[1], 3])
    
    return croped_image[0], croped_image[1]


# normalizing the images to [-1, 1]
def normalize(blur_image, sharp_image):
    blur_image = (blur_image / 127.5) - 1 
    sharp_image = (sharp_image / 127.5) - 1
    
    return blur_image, sharp_image

# normalizing the images to [-1, 1]
def normalize_single(image):
    image = (image / 127.5) - 1 
    
    return image

@tf.function()
def random_jitter(blur_image, sharp_image, input_shape): 
    # random_crop to input_shape
    blur_image, sharp_image = random_crop(blur_image, sharp_image, input_shape)
#     # reshape
#     blur_image, sharp_image = resize_img(blur_image, sharp_image, input_shape)
    
    if tf.random.uniform(()) > 0.5:
        blur_image = tf.image.flip_left_right(blur_image)
        sharp_image = tf.image.flip_left_right(sharp_image)
        
    return blur_image, sharp_image

def load_img_train(path_A, path_B, input_shape=input_shape):
    blur_image, sharp_image = load_img(path_A, path_B)
    blur_image, sharp_image = random_jitter(blur_image, sharp_image, input_shape)
    blur_image, sharp_image = normalize(blur_image, sharp_image)
    
    return blur_image, sharp_image


def make_train_dataset(root_A, root_B):

    A_files = os.listdir(root_A)
    B_files = os.listdir(root_B)
    
    A_paths = []
    B_paths = []
    for file in A_files:
        if file in B_files:
            A_paths.append(os.path.join(root_A, file))
            B_paths.append(os.path.join(root_B, file))
            
    print('nums of train data: ', len(A_paths))
    print(A_paths[0])    

    #生成Dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((A_paths, B_paths))

    dataset = dataset.map(load_img_train,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(A_paths)).batch(batch_size)    
    
    return dataset


# In[6]:


# [-1, 1] -> [0, 255]
def de_normalize(image):
    d_image = tf.cast((image + 1) * 127.5, tf.float32)

    return d_image


# In[7]:


#定义残差块函数
def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), normlization='instance'):
    initializer = tf.random_normal_initializer(0., 0.02)
    #使用步长为1的卷积，保持大小不变
    x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',
                           kernel_initializer=initializer, use_bias=False)(input)    

    if normlization == 'instance':
        x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)
    else:
        x = SwitchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dropout(0.5)(x)
    
    #再来一次步长为1的卷积
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding='same',
                           kernel_initializer=initializer, use_bias=False)(x)    
    
    if normlization == 'instance':
        x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)
    else:
        x = SwitchNormalization(axis=-1)(x)
    
    #将卷积后的结果与原始输入相加
    out = keras.layers.Add()([input, x])#残差层
    return out


# In[8]:


def Generator(normlization='instance'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inputs = keras.layers.Input(name='blur_image' ,shape=input_shape)
    
    x = keras.layers.Conv2D(filters=ngf, kernel_size=(7, 7), padding='same',
                           kernel_initializer=initializer, use_bias=False)(inputs)
    
    if normlization == 'instance':
        x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)
    else:
        x = SwitchNormalization(axis=-1)(x)
    
    x = keras.layers.Activation('relu')(x)
    
    downsampling = 2
    for i in range(downsampling):#两次下采样
        mult = 2**i
        x = keras.layers.Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)(x)
        if normlization == 'instance':
            x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)
        else:
            x = SwitchNormalization(axis=-1)(x)
        x = keras.layers.Activation('relu')(x)
        
    res_n = 2**downsampling
    for i in range(n_blocks_gen):#定义多个残差层
        x = res_block(x, ngf*res_n, normlization=normlization)
        
    for i in range(downsampling):#两次上采样
        n_f = 2**(downsampling - i)
        
        x = keras.layers.Conv2DTranspose(filters=int(ngf * n_f), kernel_size=(3, 3), strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False)(x)
#         x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(filters=int(ngf * n_f / 2), kernel_size=(3, 3), padding='same',
                                kernel_initializer=initializer, use_bias=False)(x)
        if normlization == 'instance':
            x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)
        else:
            x = SwitchNormalization(axis=-1)(x)
        x = keras.layers.Activation('relu')(x)
        
    #步长为1的卷积操作
    x = keras.layers.Conv2D(filters=output_nc, kernel_size=(7, 7), padding='same',
                            kernel_initializer=initializer, use_bias=False)(x)
    x = keras.layers.Activation('tanh')(x)
    
    outputs = keras.layers.Add()([x, inputs])#与最外层的输入完成一次大残差
    #防止特征值域过大，进行除2操作（取平均数残差）
    outputs = keras.layers.Lambda(lambda z: z/2.0)(outputs)
    #构建模型
    model = keras.Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


# In[9]:


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(keras.layers.Conv2D(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(keras.layers.LeakyReLU())

    return result


def Discriminator(norm_type='instancenorm', use_sigmoid=False):
    """Build discriminator architecture."""
    n_layers = 3
    inputs = keras.layers.Input(shape=input_shape, name='input_image')

    x = keras.layers.Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = keras.layers.LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = keras.layers.Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        
        if norm_type.lower() == 'batchnorm':
            x = keras.layers.BatchNormalization()(x)
        elif norm_type.lower() == 'instancenorm':
            x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                                gamma_initializer="random_uniform")(x)
        x = keras.layers.LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = keras.layers.Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)

    if norm_type.lower() == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    elif norm_type.lower() == 'instancenorm':
        x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform",
                                            gamma_initializer="random_uniform")(x)    
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = keras.layers.Activation('sigmoid')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='tanh')(x)
    x = keras.layers.Dense(1)(x)
    
    if use_sigmoid:
        x = keras.layers.Activation('sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=x, name='Discriminator')
    return model


def VGG_19(weights_path, tag='custom'):
    if tag != 'custom':
        vgg_19 = keras.applications.VGG19(include_top=False, weights=weights_path)
    #     vgg_19.load_weights('./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        model = keras.models.Model(inputs=vgg_19.input, outputs=vgg_19.get_layer('block3_conv3').output, name='VGG-19') # block5_conv4
        model.trainable = False
    else:
        img_input = keras.layers.Input(shape=[None, None, 3])
        # Block 1
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    #     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        model = keras.Model(img_input, x, name='vgg_19')
        model.load_weights(weights_path)
        model.trainable = False

    return model


# In[10]:


loss_object = keras.losses.BinaryCrossentropy(from_logits=True)
# 生成器损失
def dcgan_d_loss(dis_real_pred, dis_fake_pred):
    real_d_loss = loss_object(tf.ones_like(dis_real_pred), dis_real_pred)
    fake_d_loss = loss_object(tf.ones_like(dis_fake_pred), dis_fake_pred)
    return (real_d_loss + fake_d_loss)*0.5

# def wasserstein_loss(y_pred_fake, y_pred_real):
#     w_loss_fake = tf.reduce_mean(tf.math.multiply(y_pred_fake, -tf.ones_like(y_pred_fake)))
#     w_loss_real = tf.reduce_mean(tf.math.multiply(y_pred_real, tf.ones_like(y_pred_real)))
#     return w_loss_fake, w_loss_real

def wasserstein_loss(y_pred_fake, y_pred_real):
    w_loss = 0.5*tf.math.reduce_mean(y_pred_fake - y_pred_real)

    return w_loss


# In[11]:



# @tf.function
def generator_loss(disc_generated_output, gen_output, target, vgg_model, mode='wgan_gp'):
    if mode == 'dcgan':
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    else:
    #     gan_loss = tf.reduce_mean(tf.math.multiply(disc_generated_output, tf.ones_like(disc_generated_output)))
        gan_loss = -tf.math.reduce_mean(disc_generated_output)

    # mean absolute error
    gen_output = de_normalize(gen_output)
    target = de_normalize(target)
    
    gen_output_ = keras.applications.vgg19.preprocess_input(gen_output)
    target_ = keras.applications.vgg19.preprocess_input(target)
    
    perceptual_fake = vgg_model(gen_output_, training = False)
    perceptual_real = vgg_model(target_, training = False)
    perceptual_loss = tf.math.reduce_mean(tf.math.square(perceptual_real - perceptual_fake))
    
    if mode == 'dcgan':
        total_gen_loss = gan_loss + perceptual_loss
    else:
        total_gen_loss = gan_loss + (LAMBDA * perceptual_loss)

    return total_gen_loss, gan_loss, perceptual_loss


# In[12]:


optimizer_g = keras.optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)
optimizer_d = keras.optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)

# optimizer_g = keras.optimizers.RMSprop(learning_rate=1e-4)
# optimizer_d = keras.optimizers.RMSprop(learning_rate=2e-5)  # 2e-5


# In[13]:


# @tf.function
def train_step(blur_image, sharp_image, epoch, generator, discriminator, vgg, d_step=5, mode='wgan_gp'):

    for step in range(d_step):
        with tf.GradientTape() as disc_tape:

            generator_image = generator(blur_image, training = False) 
            dis_real_pred = discriminator(sharp_image, training = True)
            dis_fake_pred = discriminator(generator_image, training = True)

            w_loss = wasserstein_loss(dis_fake_pred, dis_real_pred)

            # dtype caused disconvergence?
            t = tf.random.uniform([batch_size, 1, 1, 1], minval=0.,
                                  maxval=1., dtype=tf.float32)
            x_hat = t * sharp_image + (1. - t) * generator_image

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(x_hat)
                Dx = discriminator(x_hat, training=True)

            grads = gp_tape.gradient(Dx, x_hat)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gp_loss = tf.reduce_mean((slopes - 1.) ** 2)

            d_loss = w_loss + gradient_penalty_weight*gp_loss

        gradients_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(gradients_d, discriminator.trainable_variables))
        
    with tf.GradientTape() as gen_tape:
        generator_image = generator(blur_image, training = True) 
        dis_fake_pred = discriminator(generator_image, training = True)

        gan_loss = -tf.math.reduce_mean(dis_fake_pred)
        
        # mean absolute error
#         generator_image = de_normalize(generator_image)
#         sharp_image = de_normalize(sharp_image)

#         generator_image = keras.applications.vgg19.preprocess_input(generator_image)
#         sharp_image = keras.applications.vgg19.preprocess_input(sharp_image)
#         generator_image = keras.applications.vgg16.preprocess_input(generator_image)
#         sharp_image = keras.applications.vgg16.preprocess_input(sharp_image)
        
#         g_loss, gan_loss, perceptual_loss = generator_loss(dis_fake_pred, generator_image, sharp_image, vgg, mode=mode) 

        perceptual_fake = vgg(generator_image, training = False)
        perceptual_real = vgg(sharp_image, training = False)

        # perceptual_loss = tf.math.reduce_mean(tf.math.abs(perceptual_real - perceptual_fake))
        perceptual_loss = tf.math.reduce_mean(tf.math.square(perceptual_real - perceptual_fake))

        g_loss = gan_loss + (LAMBDA * perceptual_loss)
    
    gradients_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables))
    
    with summary_writer.as_default():
        tf.summary.scalar('d_loss', d_loss, step=epoch)
        tf.summary.scalar('w_loss', w_loss, step=epoch)
        tf.summary.scalar('gp_loss', gp_loss, step=epoch)
        tf.summary.scalar('g_loss', g_loss, step=epoch)
        tf.summary.scalar('gan_loss', gan_loss, step=epoch)
        tf.summary.scalar('perceptual_loss', perceptual_loss, step=epoch)

    if mode == 'wgan_gp':
        return d_loss, w_loss, gp_loss, g_loss, gan_loss, perceptual_loss
    #     return d_loss, w_loss, gp_loss
    else:
        return d_loss, d_loss, d_loss, g_loss, gan_loss, perceptual_loss


# In[14]:


def train(train_dataset, epochs, test_path, mode='wgan_gp'):
    
    generator = Generator()
    discriminator = Discriminator()
    
#     vgg = VGG_19(weights_path='./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', tag='applications')
#     vgg_19 = keras.applications.VGG19(include_top=False, weights='./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
#     vgg = keras.Model(inputs=vgg_19.input, outputs=vgg_19.get_layer("block3_conv3").output) # block5_conv4
    vgg_16 = keras.applications.VGG16(include_top=False, weights='./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg = keras.Model(inputs=vgg_16.input, outputs=vgg_16.get_layer("block3_conv3").output) # block5_conv4
    
    vgg.trainable = False

    checkpoint = tf.train.Checkpoint(optimizer_g=optimizer_g,
                                     optimizer_d=optimizer_d,
                                     generator=generator,
                                     discriminator=discriminator)

    visual_dir="./demo/"
    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir) 
    
    for epoch in range(epochs):
        
        start = time.time()
        
        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in train_dataset.enumerate():
            d_loss, w_loss, gp_loss, g_loss, gan_loss, perceptual_loss = train_step(input_image, target, epoch, generator, discriminator, vgg, mode=mode)
        
            print('\repoch:{}-steps:{}  d_loss: {:.4f}, w_loss: {:.4f}, gp_loss: {:.4f}, g_loss: {:.4f}, gan_loss: {:.4f}, perceptual_loss: {:.4f}'.format(epoch+1, n+1,
                                                                                                d_loss, w_loss, gp_loss, g_loss, gan_loss, perceptual_loss), end = '')
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('  epoch spend time: {:.4f}'.format((time.time() - start)))
#         test_image = cv2.imread(test_path)
#         generate_images(generator, test_image, epoch)
#         print('Generator result saved')
        
    checkpoint.save(file_prefix = checkpoint_prefix)


# In[15]:


A = './data/train/A/'
B = './data/train/B/'
dataset = make_train_dataset(A, B)

train(dataset, 2, './test.png', mode='wgan_gp')  # 'wgan_gp'  'dcgan'  'wgan'

