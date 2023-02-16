import time
from ops_sequential import *
from utils import *

from tensorflow.python.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch, AUTOTUNE


class SubNetwork(tf.keras.Model):
    def __init__(self, channels, name, args):
        super(SubNetwork, self).__init__(name=name)
        self.channels = channels
        self.model = self.architecture()

        # for ResNet
        self.res_n = args.res_n
        self.label_dim = args.label_dim
    
    def architecture(self):

        residual_list = get_residual_layer(self.res_n)
        model = []

        model += [Conv(self.channels, kernel=3, strid=1, name='conv')]

        for idx, layers in enumerate(residual_list):
            for i in range(layers):
                model += [ResBlock(self.channels*(2**idx), name='resblock'+str(idx)+'_'+str(i))]
        
        model += [BatchNorm(name='batch_norm')]
        model += [Relu()]

        model += [Global_Avg_Pooling()]
        model += [FullyConnected(units=self.label_dim, sn=False, name='fc')]

        model = Sequential(model, name='resnet')

        return model
    
    def call(self, x_init, training=None, mask=None):
        x = self.model(x_init, training=training)

        return x

    def build_summary(self, input_shape, detail_summary=False):
        x_init = tf.keras.layers.Input(input_shape, name='sub_network_input')

        x = self.model(x_init)

        if detail_summary:
            self.model.summary()

        self.build_model = tf.keras.Model(x_init, x, name=self.name)
        self.build_model.summary()

    def count_parameter(self):
        x = self.model.count_params()
        return x



class Network(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir

        self.save_freq = args.save_freq

        # Set parameter
        self.dataset_name = args.dataset
        self.res_n = args.res_n
        self.augment_flag = args.augment_flag
        self.batch_size = args.batch_size
        self.iteration = args.iteration

        # Dataset
        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        self.init_lr = args.lr

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)


    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
            img_class.preprocess()
            dataset_num = len(img_class.dataset)

            print("Dataset number : ", dataset_num)

            img_slice = tf.data.Dataset.from_tensor_slices(img_class.dataset)

            img_slice = img_slice. \
                apply(shuffle_and_repeat(dataset_num)). \
                apply(map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=AUTOTUNE,
                                    drop_remainder=True))

            self.dataset_iter = iter(img_slice)

            """ Network """
            self.resnet = SubNetwork(channels=32, name='resnet18', res_n=self.res_n, label_dim=self.label_dim)

            """ Optimizer """
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.init_lr, momentum=0.9, weight_decay=1e-4)

            """ Summary """
            # mean metric
            self.loss_metric = tf.keras.metrics.Mean('loss',
                                                     dtype=tf.float32)  # In tensorboard, make a loss to smooth graph

            # print summary
            input_shape = [self.img_height, self.img_width, self.img_ch]
            self.resnet.build_summary(input_shape)

            """ Count parameters """
            params = self.resnet.count_parameter()
            print("Total network parameters : ", format(params, ','))

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(classifier=self.resnet, optimizer=self.optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """

            """ Network """
            self.resnet = SubNetwork(channels=32, name='resnet18', res_n=self.res_n, label_dim=self.label_dim)

            """ Summary """
            input_shape = [self.img_height, self.img_width, self.img_ch]
            self.resnet.build_summary(input_shape)

            """ Count parameters """
            params = self.resnet.count_parameter()
            print("Total network parameters : ", format(params, ','))

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(classifier=self.resnet)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')


    ##################################################################################
    # Train
    ##################################################################################
    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))