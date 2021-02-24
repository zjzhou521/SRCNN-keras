# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam;
from tensorflow.keras.layers import concatenate, Input, Conv2D, Conv2DTranspose
# from tensorflow.python.keras.optimizers import Adam;
# from tensorflow.python.keras.models import Model;
# from tensorflow.python.keras.layers.merge import concatenate;
# from tensorflow.python.keras.engine.input_layer import Input;
# from tensorflow.python.keras.layers.convolutional import Conv2D;
# from tensorflow.python.keras.layers.convolutional import Conv2DTranspose;

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
from modules.models import RRDB_Model
from modules.lr_scheduler import MultiStepLR
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)
import time


def load_dataset(cfg, key, shuffle=True, buffer_size=10240):
    """load dataset"""
    dataset_cfg = cfg[key]
    logging.info("load {} from {}".format(key, dataset_cfg['path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=dataset_cfg['path'],
        batch_size=cfg['batch_size'],
        gt_size=cfg['gt_size'],
        scale=cfg['scale'],
        shuffle=shuffle,
        using_bin=dataset_cfg['using_bin'],
        using_flip=dataset_cfg['using_flip'],
        using_rot=dataset_cfg['using_rot'],
        buffer_size=buffer_size)
    return dataset



def SRDenseNetBlock(inputs, i, nlayers):
    logits = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu",use_bias=True, name="conv2d_%d_%d" % (i + 1, 0 + 1))(inputs);
    for j in xrange(1, nlayers):
        middle = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu",use_bias=True, name="conv2d_%d_%d" % (i + 1, j + 1))(logits);
        logits = concatenate([logits, middle], name="concatenate_%d_%d" % (i + 1, j + 1));
    return logits;

def SRDenseNetKeras(size, channels, nblocks=8, nlayers=8, name="SRDenseNet"):
    x = inputs = Input([size, size, channels], name='input_image')
    logits = Conv2D(filters=16, kernel_size=3, strides=1,padding='same', activation="relu", use_bias=True)(x);
    gggggg = logits;
    # 8 dense blocks
    for i in xrange(nblocks):
        logits = SRDenseNetBlock(logits, i, nlayers);
        logits = concatenate([logits, gggggg]);
    # bottleneck layer
    logits = Conv2D(filters=256, kernel_size=1, padding='same',
                activation="relu", use_bias=True)(logits);
    # 2 deconv layers
    logits = Conv2DTranspose(filters=256, kernel_size=3, strides=2,
            padding='same', activation="relu", use_bias=True)(logits);
    logits = Conv2DTranspose(filters=256, kernel_size=3, strides=2,
            padding='same', activation="relu", use_bias=True)(logits);
    # reconstruction layer
    logits = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=True)(logits);
    # optimizer and loss
    # Model(inputs, logits).compile(optimizer=Adam(lr=0.00001),loss='mean_squared_error',metrics=['mean_squared_error']);
    return Model(inputs, logits, name=name);

def main():
    flags.DEFINE_string('cfg_path', './configs/train.yaml', 'config file path')
    flags.DEFINE_string('gpu', '1', 'which gpu to use')
    print model.summary()
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network (Generator)
    model = SRDenseNetKeras((cfg['input_size'], cfg['ch_size'])
    model.summary(line_length=150)

    # load dataset with shuffle
    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=True)

    # define Adam optimizer
    learning_rate = MultiStepLR(cfg['lr'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=cfg['adam_beta1_G'],
                                         beta_2=cfg['adam_beta2_G'])

    # define loss function
    mean_squared_loss_fn = tf.keras.losses.MeanSquaredError()

    # load checkpoint
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=20)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        # get output and loss
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)
            total_loss = mean_squared_loss_fn(hr, sr)
        # optimizer
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    summary_writer = tf.summary.create_file_writer('./logs')
    # prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)
    cnter = remain_steps
    # start training
    for lr, hr in train_dataset.take(remain_steps):
        cnter -= 1
        t_start = time.time()
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        total_loss = train_step(lr, hr)
        # visualize
        # prog_bar.update("loss={:.4f}, lr={:.1e}".format(total_loss.numpy(), optimizer.lr(steps).numpy()))
        stps_epoch = int(cfg['train_dataset']['num_samples']/cfg['batch_size'])
        t_end = time.time()
        print("epoch=%3d step=%4d/%d loss=%3.4f lr=%.5f stp_time=%.3f cnter=%6d"%(int(steps/stps_epoch),int(steps%stps_epoch),stps_epoch,total_loss.numpy(),optimizer.lr(steps).numpy(),t_end-t_start,cnter))
        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)
        # save checkpoint
        if(steps % stps_epoch == 0):
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))
    print("\n[*] training done!")

if __name__ == "__main__":
    main()
















