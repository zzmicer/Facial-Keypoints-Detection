from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten,Conv2DTranspose,Input,Add
from keras.layers.advanced_activations import LeakyReLU
from src.models.utils import conv_block

def build_fcn(n_classes, img_size, model_name):
    """
    Fully Convolutional Network
    
    Encode->Bottleneck->Decode
    Used FCN8-like structure, with reduced num of Conv blocks
    """
    input = Input((img_size,img_size,3))
    
    # Encode
    #Block1, output(48, 48, 64)
    x = conv_block(input,n_filters=64)
    
    #Block2, output(24, 24, 128)
    x = conv_block(x,n_filters=128)
    
    #We will fuse output of these layers before upsampling
    #Block3, output(12, 12, 512)  
    pool3 = conv_block(x,n_filters=256)
    
    #Block4, output(6, 6, 512) 
    pool4 = conv_block(pool3,n_filters=512)
    
    #Block5, output(3, 3, 512) 
    x = conv_block(pool4,n_filters=512)
    
    #Bottleneck
    #Block6, output(3, 3, 512) 
    x = Convolution2D(512,kernel_size=(1,1),padding='same',use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    #Block7, output(3, 3, 14) 
    x = Convolution2D(n_classes,kernel_size=(1,1),padding='same',use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    #Decode/UpSampling
    #Block8, output(12, 12, 14)
    output3pool = Convolution2D(n_classes,kernel_size=(1,1),padding='same',use_bias=False)(pool3)
    
    #Block9, output(6, 6, 14)
    output4pool = Convolution2D(n_classes,kernel_size=(1,1),padding='same',use_bias=False)(pool4)
    
    #Block10, output(12, 12, 14)
    up_pool4 = Conv2DTranspose(n_classes,kernel_size=(2,2),strides=2)(output4pool)
    
    #Block11, output(12, 12, 14)
    up_last_conv = Conv2DTranspose(n_classes,kernel_size=(4,4),strides=4)(x)
    
    #Block12, output(12, 12, 14)
    fusion = Add()([output3pool,up_pool4,up_last_conv])
    
    #Block13, output(96, 96, 14)
    output = Conv2DTranspose(n_classes,kernel_size=(8,8),strides=8,padding='same')(fusion)
    
    model = Model(input,output,name=model_name)
    return model

