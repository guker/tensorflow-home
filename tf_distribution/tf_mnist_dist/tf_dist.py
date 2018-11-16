# encoding:utf-8
import math
import tensorflow as tf
import tempfile
import time
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
assert tf.__version__ == '1.12.0', ('This code was tested  successly in tensorflow v1.12.0, the other version no test')

flags = tf.app.flags
IMAGE_PIXELS = 28

flags.DEFINE_string('data_dir', './mnist-data', 'Directory for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 300000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 128, 'Training batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_boolean('use_gpu', False, 'use gpu to accelate computing ')
flags.DEFINE_integer('gpu_id', 0, 'multi-gpu server')
flags.DEFINE_string('checkpoint', './checkpoint', 'checkpoint path')
flags.DEFINE_string('logdir', './log', 'Training log')

flags.DEFINE_string('ps_hosts', '192.168.32.73:2260', 'comma-seprated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '192.168.32.73:2261',
                    'comma-separated list of hostname:port pairs')

flags.DEFINE_string('job_name', None, 'job name: worker or ps')
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
flags.DEFINE_boolean('sync_replicas', False, 'Use the sync_replicas (synchronized replicas) mode')

FLAGS = flags.FLAGS


def set_learning_rate(learning_rate):
    '''
    tf.train.piecewise_constant 分段常数衰减
    tf.train.polynomial_decay   多项式衰减
    tf.train.exponential_decay  指数衰减
    tf.train.natural_exp_decay  自然指数衰减
    tf.train.cosine_decay       余弦衰减
    tf.train.linear_cosine_decay 线性余弦衰减
    ...
    '''
    pass


def set_optimizer(optimizer, learning_rate):
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
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index')
    else:
        print('task_inex: %s' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    num_worker = len(worker_spec)

    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})

    kill_ps_queue = create_done_queue(num_worker)

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    print(server.target)
    if FLAGS.job_name == 'ps':
        # server.join()

        with tf.Session(server.target) as sess:
            for i in range(num_worker):
                sess.run(kill_ps_queue.dequeue())
        return
    # worker的主节点(master), 即task_index为0的节点
    is_chief = (FLAGS.task_index == 0)
    if FLAGS.use_gpu:
        worker_device = '/job:worker/task:%d/gpu:%d' % (FLAGS.task_index, FLAGS.gpu_id)
    else:
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
    # 使用tf.train.replica_device_setter将涉及的变量分配到参数服务器上，使用cpu，
    # 如有多个参数服务器，会把变量进行轮流循环分配，即第一个变量分配到ps1，第二个变量分配到ps2，第三个分配到ps1上，循环进行
    # 将非变量的操作分配到工作节点上
    with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device='/job:ps/cpu:0',
            cluster=cluster)):

        global_step = tf.Variable(0, name='global_step', trainable=False)
        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS * IMAGE_PIXELS))
        y_ = tf.placeholder(tf.float32, shape=(None, 10))

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        tf.summary.scalar('loss', cross_entropy)

        # opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        opt = set_optimizer('Adam', FLAGS.learning_rate)

        if FLAGS.sync_replicas:
            replicas_to_aggregate = num_worker
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_worker,
                use_locking=False,
                name='mnist_sync_replicas')

        train_step = opt.minimize(cross_entropy, global_step)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                # 所有进行计算的工作节点里一个主工作节点（chief
                # 主节点负责初始化参数，模型保存，以及概要保存
                local_init_op = opt.chief_init_op
            ready_for_local_init_op = opt.ready_for_local_init_op
            # 同步训练模式所需的初始令牌，主队列
            chief_queue_runner = opt.get_chief_queue_runner()
            init_token_op = opt.get_init_tokens_op(0)

        init_op = tf.global_variables_initializer()
        kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.logdir)
        saver = tf.train.Saver()

        # train_dir = tempfile.mkdtemp()
        # 创建一个监管程序，用于统计记录训练模型中的信息
        # 主节点(cheif)负责模型参数初始化工作，此过程中，其他工作节点等待主节点完成初始化工作，一旦初始化完成，便开始训练数据
        # logdir是保存和加载模型路径，启动就会从该目录下看是否有检查点文件，若有就自动加载
        # global_step值是所有计算节点共享的
        if FLAGS.sync_replicas:
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.checkpoint,
                                     init_op=init_op, local_init_op=local_init_op,
                                     ready_for_local_init_op=ready_for_local_init_op,
                                     summary_op=summary_op, saver=saver, summary_writer=summary_writer,
                                     recovery_wait_secs=1, global_step=global_step)
        else:
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.checkpoint,
                                     init_op=init_op, recovery_wait_secs=1,
                                     summary_op=summary_op, saver=saver, summary_writer=summary_writer,
                                     global_step=global_step)

        # 创建会话，设置属性
        # 所有操作默认使用被指定的设置，如果该操作函数没有GPU实现，自动使用cpu设备
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index]
        )

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
        # 主工作节点(chief),task_index为0的节点初始化会话
        # 其他工作节点等待会话被初始化后进行计算
        # prepare_or_wait_for_session需要参数初始化完成且主节点准备好，才开始训练
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        print('Worker %d: Session initialization complete.' % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            sess.run(init_token_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        time_begin = time.time()

        print('Training begins @ %f' % time_begin)
        # 执行分布式模型训练
        local_step = 0

        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, summary_res, step = sess.run([train_step, summary_op, global_step], feed_dict=train_feed)
            sv.summary_writer.add_summary(summary_res, step)
            print('*' * 50)

            local_step += 1

            now = time.time()
            print('%f: Worker %d: training step %d done (global_step: %d)' % (now, FLAGS.task_index, local_step, step))

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print('Training ends @ %f' % time_end)

        train_time = time_end - time_begin
        print('Training elapsed time: %fs' % train_time)

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_next = sess.run(cross_entropy, feed_dict=val_feed)
        print('after %d training step(s), validation cross entropy = %g' % (FLAGS.train_steps, val_next))
        sess.run(kill_ps_enqueue_op)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
