import tensorflow as tf
from PIL import Image
import numpy as np
from core import resnet_model
from core import rnn_model
import matplotlib.pyplot as plt
import skimage.transform
import time
import os 
from core.utils import *
from core.bleu import evaluate
from core.get_data import CocoDataset
from scipy import ndimage
from scipy import ndimage, misc
class CaptionModel():
    def __init__(self, dim_images, feature_maps_layer, 
            resnet_size, n_time_step=16, **kwargs):
        '''
            Params:
                dim_images: (image_height, image_width)
                feature_maps_layer: the layer wherer features map from,can be 
                    'block_layer1','block_layer2','block_layer3' or 'block_layer4'

        '''
        self.images_batch = tf.placeholder(tf.float32, shape=(None, dim_images[0], dim_images[1], 3))
        self.captions_batch = tf.placeholder(tf.int32, [None, n_time_step + 1])
        self.dataset = CocoDataset()
        self.word_to_idx = self.dataset.word_to_idx
        self.idx_to_word = self.dataset.idx_to_word
        # self.resnet_size = resnet_size
        # self.feature_maps_layer = feature_maps_layer
        self.resnet = resnet_model.Resnet(
            self.images_batch, 
            get_layer = feature_maps_layer,
            resnet_size=resnet_size, 
            data_format='channels_first')

        self.caption_generator = rnn_model.CaptionGenerator( 
                    word_to_idx = self.word_to_idx,
                    idx_to_word = self.idx_to_word,
                    features_generator_model=self.resnet, 
                    captions = self.captions_batch, 
                    dim_embed=512, dim_hidden=1024, n_time_step=n_time_step, 
                    prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True)
        self.start_epoch = -1
        self.learning_rate = tf.placeholder(tf.float32)
        self.model_path = os.path.join('./model/', 'resnet_'+str(resnet_size), feature_maps_layer)
        self.log_path = os.path.join('./log/', 'resnet_'+str(resnet_size), feature_maps_layer)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.init_learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        
        
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum = 0.9)
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)  
        elif self.update_rule == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            # https://github.com/tensorflow/tensorflow/issues/6220
            loss = self.caption_generator.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.caption_generator.build_sampler(max_len=20)
        
        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer

            names =[var.name for var in tf.trainable_variables()]
            print('='*15, 'trainable variables:', '='*15)
            for name in names:
                print(name)
            print('='*50)

            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            # when training, the moving_mean and moving_variance need to be updated. 
            # By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, 
            # so they need to be added as a dependency to the train_op.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # train_op = optimizer.minimize(loss)
                train_op = self.optimizer.minimize(loss)
            
            # grads = tf.gradients(loss, tf.trainable_variables())
            # print(grads)
            # grads_and_vars = list(zip(grads, tf.trainable_variables()))
            # train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op   
        tf.summary.scalar('batch_loss', loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name+'/gradient', grad)
        
        summary_op = tf.summary.merge_all()

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            try:
                ckpt_state = tf.train.get_checkpoint_state(self.model_path)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)

            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model yet at %s', self.model_path)

                if self.pretrained_model is not None:
                    tf.logging.info("Start training with pretrained Model..", self.pretrained_model)
                    saver.restore(sess, self.pretrained_model)
                else:
                    tf.logging.info('No pretrained model %s', self.pretrained_model)
                    tf.logging.info('Training from scratch...')
            else:
                tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                self.start_epoch = int(ckpt_state.model_checkpoint_path.split('-')[-1])
                

            prev_loss = -1
            curr_loss = 0

            training_data_size = self.dataset.get_data_size(split='train')
            print('training_data_size:', training_data_size)

            n_iters_per_epoch = int(np.ceil(float(training_data_size)/self.batch_size))
            print('n_iters_per_epoch:', n_iters_per_epoch)
            # n_iters_per_epoch = 10
            start_t = time.time()

            for e in range(self.start_epoch+1, self.n_epochs):
                print('*'*30, 'Epochs:', e, '/', self.n_epochs, '*'*30)
                if e <= 0:
                    dynamic_learning_rate = self.init_learning_rate
                elif e <= 1:
                    dynamic_learning_rate = 0.1 * self.init_learning_rate
                elif e <= 2:
                    dynamic_learning_rate = 0.01 * self.init_learning_rate
                else:
                    dynamic_learning_rate = 0.001 * self.init_learning_rate

                for i in range(n_iters_per_epoch):
                    
                    start_i = time.time()
                    images_batch, captions_batch, _ = self.dataset.get_data_batch(
                        epoch=e, 
                        iters_in_epoch = i,
                        batch_size=self.batch_size, 
                        split = 'train')
                    time_get_data = time.time() - start_i
                    
                    # dynamic_learning_rate = 10**(-2*(e+i/n_iters_per_epoch)) * self.init_learning_rate
                    # print('dynamic_learning_rate:', dynamic_learning_rate)

                    feed_dict = {self.images_batch: images_batch, 
                                    self.captions_batch: captions_batch,
                                    self.learning_rate: dynamic_learning_rate}

                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    time_iter = time.time() - start_i
                    print('iter:', i,'/', n_iters_per_epoch, ':',
                        'get_data_time:', time_get_data, ' ',
                        'total_time:', time_iter, ' ',
                        'loss:', l)

                    # write summary for tensorboard visualization
                    if i % 20 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)
                        print('write summary:', e*n_iters_per_epoch + i, ' ', 'epoch:',e)

                    if (i+1) % self.print_every == 0:
                        print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l))

                        ground_truths = captions_batch
                        gt_decoded = decode_captions(ground_truths, self.idx_to_word)

                        gen_caps = sess.run(generated_captions, feed_dict)
                        gen_decoded = decode_captions(gen_caps, self.idx_to_word)

                        for j in range(len(gt_decoded)):
                            print("Ground truth-", j,':', gt_decoded[j])
                            print("Generated caption-", j,':', gen_decoded[j])
                            print('-'*80)                  
                        

                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                val_example_nums = self.dataset.get_images_size(split='val')
                # val_example_nums = 32
                n_iters_val = int(np.ceil(float(val_example_nums)/self.batch_size))
                
                if self.print_bleu:
                    print('*'*30, 'print bleu', '*'*30)
                    all_gen_cap = np.ndarray((val_example_nums, 20))
                    for i in range(n_iters_val):
                        print('print bleu, iter:', i)
                        val_images_batch, _, _ = self.dataset.get_data_batch(
                            batch_size=self.batch_size, 
                            iters_in_epoch = i,
                            split = 'val')
                        feed_dict = {self.images_batch: val_images_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                    
                    all_decoded = decode_captions(all_gen_cap, self.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)


                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e)
                    print("model-%s saved." %(e))




    def test(self, dataset_split = 'test', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        # features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.caption_generator.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            tf.logging.info('Restore model %s', self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            

            if attention_visualization:
                images_batch, captions_batch, images_batch_file = self.dataset.get_data_batch(
                batch_size=self.batch_size, 
                iters_in_epoch = 11,
                split = dataset_split)
                feed_dict = { self.images_batch: images_batch }
                alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
                decoded = decode_captions(sam_cap, self.idx_to_word)
                for n in range(10):
                    print("Sampled Caption: %s" %decoded[n])

                    # Plot original image
                    img = ndimage.imread(images_batch_file[n])
                    # img = images_batch[n]
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.text(0, 1, '%s'%(decoded[n]) , color='black', backgroundcolor='white', fontsize=8)
                    # Plot images with attention weights 
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(self.caption_generator.map_height,self.caption_generator.map_width)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                split_example_nums = self.dataset.get_images_size(split=dataset_split)
                all_sam_cap = np.ndarray((split_example_nums, 20))
                num_iter = int(np.ceil(float(split_example_nums) / self.batch_size))
                for i in range(num_iter):
                    print('Generating', dataset_split, 'caption_batch:', i)
                    images_batch, captions_batch, images_batch_file= self.dataset.get_data_batch(
                                                                batch_size=self.batch_size, 
                                                                iters_in_epoch = i,
                                                                split = dataset_split)
                    feed_dict = {self.images_batch: images_batch}
                    gen_cap = sess.run(sampled_captions, feed_dict=feed_dict) 
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                all_decoded = decode_captions(all_sam_cap, self.idx_to_word)
                
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(dataset_split,dataset_split))

    def test_images(self, images, images_file, captions=None, attention_visualization=True):
        '''
            Params:
                images: images batch with shape [N,H,W,C]
                captions: ground truth captions, with shape [N,max_len]
                attention_visualization: visilize the attention
        '''

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.caption_generator.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            tf.logging.info('Restore model %s', self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.images_batch: images }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.idx_to_word)

            if attention_visualization:
                for n in range(images.shape[0]):
                    print("Sampled Caption: %s" %decoded[n])

                    # Plot original image
                    img = ndimage.imread(images_file[n])
                    img = misc.imresize(img, (224, 224))
                    # img = images[n]
                    plt.subplot(5, 4, 1)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.text(0, 1, '%s'%(decoded[n]) , color='black', backgroundcolor='white', fontsize=8)
                    # Plot images with attention weights 
                    words = decoded[n].split(" ")

                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(5, 4, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(self.caption_generator.map_height,self.caption_generator.map_width)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

