# encoding=utf-8
import os
import time
import tempfile
from datetime import datetime
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

flags = tf.app.flags

flags.DEFINE_string('train_file', './train_part.txt', 'train set path list')
flags.DEFINE_string('val_file', './val_part.txt', 'validation set path list')
flags.DEFINE_integer('batch_size', 64, 'Training batch size')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_integer('num_epoches', 100, 'num_epoches')
flags.DEFINE_float('dropout', 0.5, 'dropout rate')
flags.DEFINE_integer('n_classes', 2, 'number of classes ')
flags.DEFINE_string('logdir', './log', 'Training log')
flags.DEFINE_string('checkpoint', './checkpoint', 'checkpoint path')
flags.DEFINE_integer('display_step', 1, 'display step')
flags.DEFINE_string('ps_hosts', 'localhost:2250', 'parameter server')
flags.DEFINE_string('worker_hosts', 'localhost:2251', 'worker server')
flags.DEFINE_boolean('use_gpu', False, 'use gpu to accelate computing')
flags.DEFINE_integer('gpu_id', 0, 'multi-gpu id')

flags.DEFINE_string('job_name', None, 'job_name: worker or ps')
flags.DEFINE_integer('task_index', None, 'Index of task within the job')

flags.DEFINE_boolean('sync_replicas', False, 'synchronized mode')

FLAGS = flags.FLAGS


def get_optimizer(optimizer, learning_rate):
    if optimizer == 'SGD':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'Adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'Adagrad':
        return tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'Momentum':
        return tf.train.MomentumOptimizer(learning_rate)
    elif optimizer == 'RMSProp':
        return tf.train.RMSProp(learning_rate)


def create_done_queue(num_worker):
    with tf.device("/job:ps/task:0"):
        return tf.FIFOQueue(num_worker, tf.int32, shared_name='done_queue')


def main(unused_argv):
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an expilict job_name')
    else:
        print('job_name: %s' % FLAGS.job_name)

    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index')
    else:
        print('task_index:%s' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    num_worker = len(worker_spec)

    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})

    kill_ps_queue = create_done_queue(num_worker)

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        # server.join()
        with tf.Session(server.target) as sess:
            for i in range(num_worker):
                sess.run(kill_ps_queue.dequeue())
        return

    is_chief = (FLAGS.task_index == 0)

    if FLAGS.use_gpu:
        worker_device = '/job:worker/task:%d/gpu:%d' % (FLAGS.task_index, FLAGS.gpu_id)
    else:
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index

    with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device='/job:ps/cpu:0',
            cluster=cluster)):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x')
        y = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name='y')

        keep_prob = tf.placeholder(tf.float32, name='kp')

        model = AlexNet(x, keep_prob, FLAGS.n_classes)

        score = model.fc3

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score))

        tf.summary.scalar('loss', cross_entropy)

        opt = get_optimizer('Adam', FLAGS.learning_rate)

        if FLAGS.sync_replicas:
            replicas_to_aggregate = num_worker
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_worker,
                use_locking=False,
                name='sync_replicas')

        train_op = opt.minimize(cross_entropy, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuary', accuracy)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op
            ready_for_local_init_op = opt.ready_for_local_init_op

            chief_queue_runner = opt.get_chief_queue_runner()
            init_token_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.logdir)
        saver = tf.train.Saver()

        # train_dir = tempfile.mkdtemp()

        if FLAGS.sync_replicas:
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.checkpoint,
                                     init_op=init_op, local_init_op=local_init_op,
                                     ready_for_local_init_op=ready_for_local_init_op,
                                     summary_op=summary_op, saver=saver, summary_writer=writer,
                                     recovery_wait_secs=1, global_step=global_step)
        else:
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.checkpoint,
                                     init_op=init_op, recovery_wait_secs=1,
                                     summary_op=summary_op, saver=saver, summary_writer=writer,
                                     global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index])

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        print('Worker %d: Session initialization complete.' % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            sess.run(init_token_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        train_generator = ImageDataGenerator(
            FLAGS.train_file, horizontal_flip=True, shuffle=True)
        val_generator = ImageDataGenerator(FLAGS.val_file, shuffle=False)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(train_generator.data_size /
                                           FLAGS.batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(val_generator.data_size /
                                         FLAGS.batch_size).astype(np.int16)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          FLAGS.logdir))

        for epoch in range(FLAGS.num_epoches):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            step = 1

            while step < train_batches_per_epoch:

                start_time = time.time()
                # Get a batch of images and labels

                batch_xs, batch_ys = train_generator.next_batch(FLAGS.batch_size)

                # And run the training op
                _, loss, gstep = sess.run(
                    [train_op, cross_entropy, global_step],
                    feed_dict={x: batch_xs,
                               y: batch_ys,
                               keep_prob: FLAGS.dropout})

                print('total step: %d, loss: %f' % (gstep, loss))
                duration = time.time() - start_time

                # Generate summary with the current batch of data and write to file
                if step % FLAGS.display_step == 0:
                    s = sess.run(
                        sv.summary_op,
                        feed_dict={x: batch_xs,
                                   y: batch_ys,
                                   keep_prob: 1.})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)
                # print

                if step % 10 == 0:
                    print("[INFO] {} pics has trained. time using {}".format(step * FLAGS.batch_size, duration))

                step += 1

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            test_acc = 0.
            test_count = 0
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_generator.next_batch(FLAGS.batch_size)
                acc = sess.run(
                    accuracy, feed_dict={x: batch_tx,
                                         y: batch_ty,
                                         keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("Validation Accuracy = {} {}".format(datetime.now(), test_acc))

            # Reset the file pointer of t
            # he image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(
                FLAGS.checkpoint, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = sv.saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))


if __name__ == '__main__':
    tf.app.run()
