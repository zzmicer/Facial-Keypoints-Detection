from efficientnet.keras import EfficientNetB0
from keras.models import Model
from keras.layers import Input
from src.models.utils import vanilla_head

def build_efficient_net(n_classes, img_size, model_name):
    
    input_layer = Input((img_size,img_size,3))
    eff_net = EfficientNetB0(weights=None,include_top=False, input_tensor=input_layer)
    eff_net.trainable = True

    head = vanilla_head(eff_net.output,n_classes)
    return Model(input_layer, head, name = model_name)