from src.models.utils import conv_block,vanilla_head
from keras.models import Model
from keras.layers import Input
    

def build_vanilla_cnn(n_classes, img_size, model_name):

    input_layer = Input((img_size,img_size,3))
    
    conv1 = conv_block(input_layer,n_filters=32,kernel_size=(3,3))
    conv2 = conv_block(conv1,n_filters=64,kernel_size=(3,3))
    conv3 = conv_block(conv2,n_filters=96,kernel_size=(3,3))
    conv4 = conv_block(conv3,n_filters=128,kernel_size=(3,3))
    conv5 = conv_block(conv4,n_filters=256,kernel_size=(3,3))
    conv6 = conv_block(conv5,n_filters=512,kernel_size=(3,3))

    head = vanilla_head(conv6,n_classes)
 
    return Model(input_layer, head, name = model_name)



