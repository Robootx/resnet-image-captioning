import tensorflow as tf
from PIL import Image
import numpy as np
import os
# import pickle 
import pickle
from scipy import ndimage
import matplotlib.pyplot as plt
from core.utils import *

class CocoDataset():
    def __init__(self, root_path='./data'):
        self.root_path = root_path
        
        self.splits = ['train', 'val', 'test']

        self.datas = self._read_data() # contain 3 data splits 'train', 'val', 'test', 
                                        # each split contain 'file_names', 'captions' and 'image_idxs'
        self.word_to_idx = self.get_word_to_idx()
        self.idx_to_word = self.get_idx_to_word()


    def _read_data(self):
        datas = {}
        for split in self.splits:
            data = {}
            with open(os.path.join(self.root_path, split, '%s.file.names.pkl' %split), 'rb') as f:
                data['file_names'] = pickle.load(f)   
            with open(os.path.join(self.root_path, split, '%s.captions.pkl' %split), 'rb') as f:
                data['captions'] = pickle.load(f)
            with open(os.path.join(self.root_path, split, '%s.image.idxs.pkl' %split), 'rb') as f:
                data['image_idxs'] = pickle.load(f)
            datas[split] = data

        return datas

    def get_word_to_idx(self):
        '''
            Return:
                a dict, whose key is word, value is idx
        '''
        with open(os.path.join(self.root_path, 'train', 'word_to_idx.pkl'), 'rb') as f:
            word_to_idx = pickle.load(f)
            print('Load word_to_idx with length:', len(word_to_idx))

        return word_to_idx
    def get_idx_to_word(self):
        word_to_idx = self.get_word_to_idx()
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        return idx_to_word

    def get_data_size(self, split='train'):

        return self.datas[split]['captions'].shape[0]

    def get_images_size(self, split='val'):
        
        return self.datas[split]['file_names'].shape[0]

    def get_data_batch(self, batch_size, split='train', epoch=0, iters_in_epoch=0):
        '''
            Params:
                epoch: as seed to generate same global index for one epoch
                batch_size: batch size
                dataset: 'train' or 'val'
            Returns:
                image_batch: numpy arrary with shape [batch_size, height, width, channels]
                caption_batch: numpy array with shape [batch_size, max_len] and type int
        '''
        images_batch, captions_batch, images_batch_file = None, None, None
        n_examples = self.get_data_size(split=split)
        

        
        if iters_in_epoch*batch_size > n_examples:
            print('Index out of range')
            raise 

        if split == 'train':
            np.random.seed(epoch)
            rand_idxs = np.random.permutation(n_examples)
            captions = self.datas[split]['captions'][rand_idxs]
            image_idxs = self.datas[split]['image_idxs'][rand_idxs]

            captions_batch = captions[iters_in_epoch*batch_size:(iters_in_epoch+1)*batch_size]
            image_idxs_batch = image_idxs[iters_in_epoch*batch_size:(iters_in_epoch+1)*batch_size]
            images_batch, images_batch_file = self._image_idxs_to_images(image_idxs_batch, split=split)
        elif split == 'val': #
            end = (iters_in_epoch+1)*batch_size
            if end > self.get_images_size(split=split):
                end = self.get_images_size(split=split)

            image_idxs_batch = range(iters_in_epoch*batch_size, end)

            images_batch, images_batch_file = self._image_idxs_to_images(image_idxs_batch, split=split)
        elif split == 'test': #
            end = (iters_in_epoch+1)*batch_size
            if end > self.get_images_size(split=split):
                end = self.get_images_size(split=split)

            image_idxs_batch = range(iters_in_epoch*batch_size, end)

            images_batch, images_batch_file = self._image_idxs_to_images(image_idxs_batch, split=split)
        else:
            print(split, 'not in \'train\', \'val\' or \'test\'!')
            raise 
        return images_batch, captions_batch, images_batch_file

    def image_idxs_to_captions(self, image_idxs_batch, split='val'):
        '''
            return a list of captions, each element is a variable length list of caption 
            correspond to image with index as its id
        '''
        pass


    def _image_idxs_to_images(self, image_idxs_batch, split='train'):
        '''
            Read images into batch accoring to image_idxs_batch
            Params:
                image_idxs_batch: image indexes with shape [N,]
                image_paths: all images path
            Returns:
                images_batch: numpy ndarry, images with shape [N, H, W, C]
        '''
        images_batch_file = self.datas[split]['file_names'][image_idxs_batch]
        # print('ll',images_batch_file.shape)
        images_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), images_batch_file))).astype(
                np.float32)

        return images_batch, images_batch_file





if __name__ == '__main__':
    dataset = CocoDataset()
    print(dataset.datas['train']['file_names'].shape)
    print(dataset.datas['train']['captions'].shape)
    print(dataset.datas['train']['image_idxs'].shape)
    print(dataset.get_data_size())
    images_batch, captions_batch, files_batch = dataset.get_data_batch(
        batch_size=64, 
        split='test', 
        epoch=0, 
        iters_in_epoch=0)

    print(images_batch.dtype, captions_batch.shape)

    
    idx_to_word = dataset.get_idx_to_word()
    captions_str = decode_captions(captions_batch, idx_to_word)
    for i in range(files_batch.shape[0]):
        img = ndimage.imread(files_batch[i])
    
        print(captions_str[i])
        plt.imshow(img)
        plt.axis('off')
        plt.text(0, 1, '%s'%(captions_str[i][7:]) , color='black', backgroundcolor='white', fontsize=8)
        plt.show()