import numpy as np
import tensorflow as tf


def gradient_saliency(model, data, index=None):
    data = tf.convert_to_tensor(data)
    with tf.GradientTape() as tape:
        tape.watch(data)
        pred = model(data)
    
        n_classes = get_num_classes(model)
        if index is None:
            loss = pred
        elif n_classes > 1:
            o = tf.one_hot([index], n_classes)
            loss = tf.keras.losses.categorical_crossentropy(o, pred)
        else:
            o = tf.ones_like(pred) * index
            loss = tf.keras.losses.binary_crossentropy(o, pred)
    return tape.gradient(loss, data).numpy()


def smooth_grad(model, data, index=None, noise_level=0.2, n=32):
    # SmoothGrad: https://arxiv.org/pdf/1706.03825.pdf
    if len(model.input_shape) == len(data.shape) + 1:
        data = np.expand_dims(data, 0)
    elif data.shape[0] == 1:
        pass
    else:
        raise ValueError('')

    max_min = np.max(data) - np.min(data)
    data = np.repeat(data, n, axis=0)
    noise = np.random.random(data.shape).astype('float32')
    noise = noise * noise_level * max_min
    data += noise

    return np.mean(gradient_saliency(model, data, index), axis=0)


def get_num_classes(model):
    if len(model.output_shape) == 1:
        return 1
    return model.output_shape[-1]

