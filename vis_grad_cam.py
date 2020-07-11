class arg():
    gpus = '-1'
    model = 'challenge'
    pad_size = 19
    step_size = 9
    feature = 'mel'
    skip = 1
    norm = False
    noise_aug = False
    voice_aug = False
    aug = False
    snr = ['0']
    layer = -4
    algorithm = 'cam'
    class_index = 4
    before_softmax = -1
    dataset = 'challenge'
    window = False
config = arg()
if config.model == 'challenge':
    config.before_softmax = -1
elif config.model == 'st_attention':
    config.before_softmax = -2
elif config.model == 'bdnn':
    config.before_softmax = -2
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

import pickle, scipy, os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import preprocess_spec, normalize_spec
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
import pdb

def sequence_to_windows(sequence, 
                        pad_size, 
                        step_size, 
                        skip=1,
                        padding=True, 
                        const_value=0):
    '''
    SEQUENCE: (time, ...)
    PAD_SIZE:  int -> width of the window // 2
    STEP_SIZE: int -> step size inside the window
    SKIP:      int -> skip windows...
        ex) if skip == 2, total number of windows will be halved.
    PADDING:   bool -> whether the sequence is padded or not
    CONST_VALUE: (int, float) -> value to fill in the padding

    RETURN: (time, window, ...)
    '''
    assert (pad_size-1) % step_size == 0
    window = np.concatenate([np.arange(-pad_size, -step_size, step_size),
                             np.array([-1, 0, 1]),
                             np.arange(step_size+1, pad_size+1, step_size)],
                            axis=0)
    window += pad_size
    output_len = len(sequence) if padding else len(sequence) - 2*pad_size
    window = window[np.newaxis, :] + np.arange(0, output_len, skip)[:, np.newaxis]

    if padding:
        pad = np.ones((pad_size, *sequence.shape[1:]), dtype=np.float32)
        pad *= const_value
        sequence = np.concatenate([pad, sequence, pad], axis=0)

    return np.take(sequence, window, axis=0)

def label_to_window(config, skip=1):
    def _preprocess_label(label):
        label = sequence_to_windows(
            label, config.pad_size, config.step_size, skip, True)
        return label
    return _preprocess_label



def windows_to_sequence(windows,
                        pad_size,
                        step_size):
    windows = np.array(windows)
    sequence = np.zeros((windows.shape[0],) + windows.shape[2:],
                        dtype=np.float32)   
    indices = np.arange(1, windows.shape[0]+1)
    indices = sequence_to_windows(
        indices, pad_size, step_size, True, -1)

    for i in range(windows.shape[0]):
        pred = windows[np.where(indices-1 == i)]
        sequence[i] = pred.mean(axis=0)
    
    return sequence



# @tf.function
def multipling(inputs):
    conv, weights = inputs
    # weights = tf.expand_dims(weights,0)
    grad_cam = conv * weights
    grad_cam = tf.math.reduce_sum(grad_cam, axis=-1)
    return grad_cam

# @tf.function
def generate_grad_cam(model,data,class_idx,new_model, y_model):
    # data = (sound time, window, seq)
    img_tensor = tf.convert_to_tensor(data)

    # class index별 나눠서 진행하는 것 전처리
    masking = np.zeros(data.shape[1])
    ones = np.array([1])
    def get_grad_val(inputs):
        with tf.GradientTape() as y_tape:
            y_tape.watch(img_tensor)
            y_c = y_model(img_tensor, training=False)
            class_mask = tf.one_hot(class_idx, y_c.shape[-1])
            if tf.rank(y_c) == 2:
                class_mask = class_mask[tf.newaxis, ...]
            y_c = y_c * class_mask
        y_c_grad = y_tape.gradient(y_c, img_tensor)
        with tf.GradientTape() as A_tape:
            A_tape.watch(img_tensor)
            A_k = new_model(img_tensor, training=False)
        A_k_grad = A_tape.gradient(A_k, img_tensor)
        return y_c_grad / A_k_grad, A_k
    
    def get_grad_val_window(inputs):
        y_c_grad = tf.zeros_like(inputs)
        for i in range(2**data.shape[1]):
            binary = bin(i)[2:]
            binary = '0' * (data.shape[1] - len(binary)) + binary
            if int(binary[len(binary) // 2]) != class_idx:
                continue
            for j,k in enumerate(binary):
                if k == '1':
                    masking[j] = ones
            def y_mask(inputs):
                return inputs * masking
            # 이 부분에서 class_idx 다 반영해서 y_c_grad 값 뽑도록 수정
            with tf.GradientTape() as y_tape:
                y_tape.watch(img_tensor)
                y = y_model(img_tensor, training=False) 
                y_c = tf.map_fn(y_mask, y)
            y_c_grad += y_tape.gradient(y_c, img_tensor)
            
        y_c_grad /= 2 ** (data.shape[1] - 1) 
        with tf.GradientTape() as A_tape:
            A_tape.watch(img_tensor)
            A_k = new_model(img_tensor, training=False)
            
        A_k_grad = A_tape.gradient(A_k, img_tensor)
        return y_c_grad / A_k_grad, A_k

    grad_val, conv_output = None, None
    #----------------------------------------------------------
    if config.window:
        grad_val, conv_output = get_grad_val_window(img_tensor)
    else:
        grad_val, conv_output = get_grad_val(img_tensor)
    #----------------------------------------------------------
    if grad_val.shape[0] == 1:
        grad_val = tf.squeeze(grad_val, axis=0)
    if conv_output.shape[0] == 1:
        conv_output = tf.squeeze(conv_output, axis=0)
    
    def image_resize(image, size=(data.shape[0], data.shape[1])):
        return tf.image.resize(image, size)
    if len(grad_val.shape) == 3:
        axis = (0,1)
        weights = tf.keras.backend.mean(tf.cast(grad_val, tf.float32), axis=axis)
        # 음성 길이, 모델 출력 channel
        for i in range(4-tf.rank(conv_output)):
            conv_output = conv_output[..., tf.newaxis]
        # conv_output = (time, 7, 10, 128), weights = (time,)
        cam = tf.map_fn(multipling, (conv_output, weights), dtype='float32')
        # cam = (time, 7, 10)
        
        cam = cam[..., tf.newaxis]
        cam = tf.map_fn(image_resize, cam)

        ## Relu
        cam = tf.keras.activations.relu(cam)
        cam = (cam + tf.keras.backend.abs(cam)) / 2
        cam = tf.math.divide_no_nan(tf.squeeze(cam, -1), tf.keras.backend.max(cam,axis=-1))
    
        return cam
    elif len(grad_val.shape) == 4:
        # if tf.rank(grad_val) == tf.rank(conv_output) == 4:
        #     grad_val = tf.squeeze(grad_val, axis=0)
        #     conv_output = tf.squeeze(conv_output, axis=0)
        axis = (1,2,3)
        weights = tf.keras.backend.mean(tf.cast(grad_val, tf.float32), axis=axis)

        cam = weights * tf.math.reduce_sum(grad_val,-1)
        if cam.shape[0] == 1:
            cam = tf.squeeze(cam, axis=0)
        if tf.rank(cam) == 2:
            cam = cam[...,tf.newaxis]
        cam = tf.image.resize(cam, data.shape[1:3])
        cam = tf.keras.activations.relu(cam)
        cam = (cam + tf.keras.backend.abs(cam)) / 2
        cam = tf.math.divide_no_nan(cam, tf.keras.backend.max(cam))
        return cam
    else:
        raise ValueError(f'grad_val shape is {grad_val.shape}')
    

def gradient_saliency(model, data):
    data = tf.convert_to_tensor(data)
    with tf.GradientTape() as tape:
        tape.watch(data)
        y = model(data)
    return tape.gradient(y, data).numpy()

def main(config):
    ## 2. image sources
    x, y = None, None
    if config.dataset == 'noisex':
        data_path = '/root/datasets/ai_challenge/TIMIT_noisex3'
        x = pickle.load(open(data_path + '/snr0_10.pickle', 'rb'))
        x = list(map(preprocess_spec(config, feature=config.feature), x))[:5]
        y = pickle.load(open(os.path.join(data_path, f'label_10.pickle'), 'rb'))
        y = list(map(label_to_window(config, skip=config.skip), y))
    elif config.dataset == 'challenge':
        data_path = '/root/datasets/ai_challenge/' # inside a docker container
        if not os.path.isdir(data_path): # outside... 
            data_path = '/media/data1/datasets/ai_challenge/'

        # x = np.load(os.path.join(data_path, 't3_audio.npy'))
        # x = pickle.load(open(data_path+'/final_x.pickle','rb'))
        x = pickle.load(open(data_path+'/test_final_x.pickle','rb'))
        x = normalize_spec(x, norm=config.norm)

        """ DATA """
        # 2. DATA PRE-PROCESSING
        print("data pre-processing finished")
        
    
    
    if config.dataset == 'challenge':
        H5_PATH = '/root/RDChallenge/window/the_model.h5'
        config.model = 'challenge'
        config.window = False
    else:
        H5_PATH = '/root/RDChallenge/window/TIMIT_noisex3_divideSNR/' \
            f'{config.model}_0.2_sgd_19_9_skip2_decay0.95_mel_batch4096_noiseaug_voiceaug_aug.h5'

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    model.summary()
    exit()
    new_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=model.layers[config.layer].output)
    y_model = tf.keras.models.load_model(H5_PATH, compile=False)
    y_model.layers[config.before_softmax].activation = None
    class_idx = config.class_index

    k = 0
    maps = []
    for _x in tqdm(x):
        if tf.rank(_x) == tf.rank(model.input) - 1:
            _x = _x[tf.newaxis,...]
        img = np.zeros((_x.shape[0], _x.shape[-1]))
        cam = np.zeros((_x.shape[0], _x.shape[-1]))
        tmp_cam = np.zeros_like(_x)
        y_c = new_model(_x, training=False)
        if config.algorithm == 'cam':
            ### grad-cam code ###
            _cam = generate_grad_cam(model, _x, class_idx, new_model, y_model)
            # for i, j in enumerate(s):
            #     _cam = _generate_grad_cam(np.expand_dims(j, 0),model, class_idx, config.layer)
            #     tmp_cam[i] = np.expand_dims(_cam.T, 0)
            if config.model != 'challenge':
                _img = _x
                if _cam.shape[-1] != 80:
                    _cam = tf.transpose(_cam, [0,2,1])
                if _img.shape[-1] != 80:
                    _img = np.transpose(_x, [0,2,1])

                cam = windows_to_sequence(_cam, config.pad_size, config.step_size)
                img = windows_to_sequence(_img, config.pad_size, config.step_size)
            else:
                img = _x
                cam = _cam.numpy()
            #####################

        # print(np.array(cam).shape, np.array(img).shape)
        elif config.algorithm == 'sal':
            ### saliency code ###
            cam = gradient_saliency(tf.keras.Model(inputs=model.input,outputs=model.layers[config.layer].output), _x)
            cam = windows_to_sequence(cam, config.pad_size, config.step_size)
            img = windows_to_sequence(_x, config.pad_size, config.step_size)
            #####################
        maps.append([img,cam])
    print('process done, save data')
    name = f'grad_{config.algorithm}_{config.model}_data_{config.snr[0]}_layer{config.layer}_class{config.class_index}_data{config.dataset}'
    if config.window:
        name += '_window'
    
    if config.algorithm == 'sal':
        name = name[5:]
    pickle.dump(maps, open(name+'.pickle', 'wb'))
    print(name)
    print(model.layers[config.layer].name)

if __name__ == '__main__':
    for i in range(0,11):
        config.class_index = i
        main(config)