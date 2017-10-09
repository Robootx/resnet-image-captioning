from core.model import CaptionModel
import tensorflow as tf
import os
from scipy import ndimage, misc
import numpy as np
from PIL import Image

def main():

    model = CaptionModel(
        dim_images=(224, 224), 
        feature_maps_layer='block_layer3', 
        resnet_size=50,  
        n_time_step=16,
        n_epochs=50, 
        batch_size=32, 
        update_rule='adam',
        learning_rate=0.001,    
        print_every=1000, # 1000
        save_every=1,       # save every save_every epoch
        pretrained_model=None, 
        test_model='model/resnet_50/block_layer4/model-8',
        print_bleu=False, 
        )

    # model.train()
    model.test(dataset_split = 'test', attention_visualization=False, save_sampled_captions=True)
    # demo(model=model)

def demo(model):
    img_root_dir = 'demo'
    images_batch_file = ['0.jpg', '1.jpg','2.jpg','4.jpg', '5.jpg', '6.jpg', '7.jpg']
    images_batch_file = np.array([os.path.join(img_root_dir,name) for name in images_batch_file])
    # images_batch = np.array(ndimage.imread(images_batch_file[0], mode='RGB')).astype(np.float32)
    # images_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB').resize([224, 224], Image.ANTIALIAS), images_batch_file))).astype(
    #             np.float32)
    images_batch = []
    for file_name in images_batch_file:
        image = ndimage.imread(file_name, mode='RGB')
        image = misc.imresize(image, (224, 224))
        images_batch.append(image)
    images_batch = np.array(images_batch).astype(np.float32)

    print(images_batch.shape)
    model.test_images(images=images_batch, images_file=images_batch_file)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()