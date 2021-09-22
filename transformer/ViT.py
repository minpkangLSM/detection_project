"""
2021.06.02.
"AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.", 2020.
Kangmin Park, Lab. for Sensor and Modeling, Univ. of Seoul.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LayerNormalization
from backbones.utils import sequentializer_conv, sequentializer_patch, vision_transformer
from backbones.utils import positionembds, classtoken

def ViT(image_shape,
        patch_size,
        hidden_size,
        mlp_dim,
        head_num,
        layer_num):
    assert image_shape[0]%patch_size == 0, "image shape should be a multiple of path size"

    input = tf.keras.Input(shape=image_shape)
    # method1
    seq = sequentializer_conv(input_tensor=input,
                              patch_size=patch_size,
                              embedding_dim=hidden_size)
    # # method2
    # seq = sequentializer_patch(input_tensor=input,
    #                            patch_size=patch_size,
    #                            embedding_dim=hidden_size)
    position_embd = positionembds(seq.shape)
    seq = seq + position_embd
    class_token = classtoken(seq)
    y = tf.concat([class_token, seq], axis=1)

    for i in range(layer_num):
        y = vision_transformer(input_tensor=y,
                               num_heads=head_num,
                               mlp_dim=mlp_dim)
    y = LayerNormalization()(y)
    y = tf.keras.layers.Lambda(lambda v : v[:,0], name="ExtractToken")(y)
    model = tf.keras.models.Model(inputs=input, outputs=y)
    return model

if __name__ == "__main__" :

    image_shape = (512, 512, 3)
    patch_size = 16
    hidden_size = 768
    mlp_dim = 3072
    head_num = 12
    layer_num = 12
    model = ViT(image_shape=image_shape,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                head_num=head_num,
                layer_num=layer_num)
    model.summary()







