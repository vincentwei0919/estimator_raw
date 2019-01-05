# coding:utf-8
#导入相关库
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# 加载通过TensorFlow-Silm定义好的 inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import pdb
#图片数据地址
TRAIN_DATA = '/home/dhao/03-weiwei/beauty_prediction/data/14-processed/tfrecord/male/male_train.tfrecords'
TEST_DATA = '/home/dhao/03-weiwei/beauty_prediction/data/14-processed/tfrecord/male/male_val.tfrecords'

shuffle_buffer = 10000
BATCH = 64
#打开 estimator 日志
tf.logging.set_verbosity(tf.logging.INFO)

#自定义模型
#这里我们提供了两种方案。一种是直接通过slim工具定义已有模型
#另一种是通过tf.layer更加灵活地定义神经网络结构
def inception_v3_model(image,is_training):
    saver = tf.train.Saver()
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        net,end_points = inception_v3.inception_v3(image,num_classes=None)
        with tf.variable_scope('Logits'):
            net = slim.flatten(net, scopre='flatten')
            predictions = slim.fully_connected(net, 1, activation_fn=None, scope='output')
            return predictions

#自定义Estimator中使用的模型。定义的函数有4个输入，
#features给出在输入函数中会提供的输入层张量。这是个字典
#字典通过input_fn提供。如果是系统的输入
#系统会提供tf.estimator.inputs.numpy_input_fn中的x参数指定内容
#labels是正确答案，通过numpy_input_fn的y参数给出
#在这里我们用dataset来自定义输入函数。
#mode取值有3种可能，分别对应Estimator的train,evaluate,predict这三个函数
#mode参数可以判断当前是训练，预测还是验证模式。
#最有一个参数param也是字典，里面是有关于这个模型的相关任何超参数（学习率）
def model_fn(features,labels,mode,params):
    predict = inception_v3_model(features,mode == tf.estimator.ModeKeys.TRAIN)
    #如果是预测模式，直接返回结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"result":tf.argmax(predict,1)}
        )
    #定义损失函数，这里使用tf.losses可以直接从tf.losses.get_total_loss()拿到损失
    pdb.set_trace()
    predict = tf.squeeze(predict)
    tf.losses.mean_squared_error(labels, predict)
    #优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    #定义训练过程。传入global_step的目的，为了在TensorBoard中显示图像的横坐标
    train_op = optimizer.minimize(
        loss=tf.losses.get_total_loss(),
        global_step=tf.train.get_global_step()
    )

    #定义评测标准
    #这个函数会在调用Estimator.evaluate的时候调用
    accuracy = tf.metrics.mean_squared_error(labels, predict, name='acc')
    eval_metric_ops = {
        "my_metric":accuracy
    }
    #用于向TensorBoard输出准确率图像
    #如果你不需要使用TensorBoard可以不添加这行代码
    tf.summary.scalar('accuracy', accuracy[1])
    #model_fn会返回一个EstimatorSpec
    #EstimatorSpec必须包含模型损失，训练函数。其它为可选项
    #eval_metric_ops用于定义调用Estimator.evaluate()时候所指定的函数
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=tf.losses.get_total_loss(),
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_data(record):
    # filename_queue = tf.train.string_input_producer([record])
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)

    # features = tf.parse_single_example(serialized_example, features={
    features = tf.parse_single_example(record, features={
        "label": tf.FixedLenFeature([], tf.float32),
        "image_raw": tf.FixedLenFeature([], tf.string)
    })
    imgs = tf.decode_raw(features["image_raw"], tf.uint8)
    imgs = tf.reshape(imgs, [224, 224, 3])
    imgs = tf.cast(imgs, tf.float32)
    imgs =  imgs / 255.
    imgs = imgs - tf.reduce_mean(imgs, axis=0)
    labels = tf.cast(features['label'], tf.float32)
    return imgs, labels


#从dataset中读取训练数据，这里和之前处理花朵分类的时候一样
def my_input_fn(file):
    dataset = tf.data.TFRecordDataset([file])
    dataset = dataset.map(get_data)
    dataset = dataset.shuffle(shuffle_buffer).batch(BATCH)
    dataset = dataset.repeat(10)
    iterator = dataset.make_one_shot_iterator()
    batch_img,batch_labels = iterator.get_next()
    with tf.Session() as sess:
        batch_sess_img,batch_sess_labels = sess.run([batch_img,batch_labels])
        #这里需要特别注意 由于batch_sess_img这里是转成了string后在原有长度上增加了8倍
        #所以在这里我们要先转成numpy然后再reshape要不然会报错
        # pdb.set_trace()
        # batch_sess_img = np.fromstring(batch_sess_img, dtype=np.float32)

        #numpy转换成Tensor
        # batch_sess_img = tf.reshape(batch_sess_img, [BATCH, 224, 224, 3])
    return batch_sess_img,batch_sess_labels


def main():
    #定义超参数
    model_params = {"learning_rate":0.001}
    #定义训练的相关配置参数
    #keep_checkpoint_max=1表示在只在目录下保存一份模型文件
    #log_step_count_steps=50表示每训练50次输出一次损失的值
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1,log_step_count_steps=50)
    #通过tf.estimator.Estimator来生成自定义模型
    #把我们自定义的model_fn和超参数传进去
    #这里我们还传入了持久化模型的目录
    #estimator会自动帮我们把模型持久化到这个目录下
    estimator = tf.estimator.Estimator(model_fn=model_fn,params=model_params,model_dir="./model_save",config=run_config)
    #开始训练模型，这里说一下lambda表达式
    #lambda表达式会把函数原本的输入参数变成0个或它指定的参数。可以理解为函数的默认值
    #这里传入自定义输入函数，和训练的轮数
    estimator.train(input_fn=lambda :my_input_fn(TRAIN_DATA),steps=300)
    #训练完后进行验证，这里传入我们的测试数据
    test_result = estimator.evaluate(input_fn=lambda :my_input_fn(TEST_DATA))
    #输出测试验证结果
    accuracy_score = test_result["my_metric"]
    print("\nTest accuracy:%g %%"%(accuracy_score*100))

if __name__ == '__main__':
    main()

