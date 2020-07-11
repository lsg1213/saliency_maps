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
    layer = -3
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
import tensorflow.keras.backend as K
from tqdm import tqdm
from utils import preprocess_spec, normalize_spec
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus


def multipling(inputs):
    conv, weights = inputs
    # weights = tf.expand_dims(weights,0)
    grad_cam = conv * weights
    grad_cam = tf.math.reduce_sum(grad_cam, axis=-1)
    return grad_cam

def generate_grad_cam_by_sample(model,data,class_idx):
    # data = (sound time, window, seq)
    # y_model: model에서 마지막 fc에서 softmax activation만 제거한 모델
    # new_model: model에서 내가 보고 싶은 layer까지만 잘라놓은 모델
    img_tensor = tf.convert_to_tensor(data)
    
    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    class_output = model.output[:, class_idx]

    conv_output = model.layers[config.layer].output
    grads = K.gradients(class_output, conv_output)[0]

    grad_f = K.function([model.input], [conv_output, grads])
    output, grads_val = grad_f([data])

    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0,1))
    cam = np.dot(output, weights)
    if len(cam.shape) == 2:
        cam = np.expand_dims(cam, axis=-1)
    
    return cam
    

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
        
    class_idx = config.class_index
    
    with tf.Graph().as_default():
        tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.load_model(H5_PATH, compile=False)
        k = 0
        maps = []
        for _x in tqdm(x):
            if len(_x.shape) == len(model.input.shape) - 1:
                _x = np.expand_dims(_x, axis=0)
            img = None
            cam = None
            if config.algorithm == 'cam':
                ### grad-cam code ###
                _cam = generate_grad_cam_by_sample(model, _x, class_idx)
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
                    cam = _cam
            maps.append([img,cam])
    tf.compat.v1.enable_eager_execution()
    for i, j in enumerate(maps):
        cam = tf.image.resize(j[1],(j[0].shape[1], j[0].shape[2]),antialias=True)
        maps[i][1] = cam.numpy()
    print('process done, save data')
    name = f'grad_{config.algorithm}_{config.model}_data_{config.snr[0]}_layer{config.layer}_class{config.class_index}_data{config.dataset}'
    if config.window:
        name += '_window'
    
    pickle.dump(maps, open(name+'.pickle', 'wb'))
    print(name)
    print(model.layers[config.layer].name)


if __name__ == '__main__':
    for i in range(0,11):
        config.class_index = i
        main(config)