import sys
import json
import logging
import os
import time
import psutil
import argparse
import threading
import jieba
import random

from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import data_utils
from as_reader_tf import AttentionSumReaderTf
from attention_sum_reader import AttentionSumReader

tf.app.flags.DEFINE_integer(flag_name="num_epoches_new", default_value=100, docstring="")
tf.app.flags.DEFINE_integer(flag_name="max_count", default_value=35200, docstring="")

# 基础参数
tf.app.flags.DEFINE_bool(flag_name="debug",
                         default_value=False,
                         docstring="是否在debug模式")

tf.app.flags.DEFINE_bool(flag_name="train",
                         default_value=False,
                         docstring="进行训练")

tf.app.flags.DEFINE_bool(flag_name="test",
                         default_value=False,
                         docstring="进行测试")

tf.app.flags.DEFINE_bool(flag_name="predict",
                         default_value=True,
                         docstring="进行预测")

tf.app.flags.DEFINE_bool(flag_name="ensemble",
                         default_value=False,
                         docstring="进行集成模型的测试")

tf.app.flags.DEFINE_integer(flag_name="random_seed",
                            default_value=1007,
                            docstring="随机数种子")

tf.app.flags.DEFINE_string(flag_name="log_file",
                           default_value=None,
                           docstring="是否将日志存储在文件中")

tf.app.flags.DEFINE_string(flag_name="weight_path",
                           default_value="model/",
                           docstring="之前训练的模型权重")

tf.app.flags.DEFINE_string(flag_name="framework",
                           default_value="tensorflow",
                           docstring="使用的模型框架，“tensorflow”或者“keras”")

# 定义数据源
tf.app.flags.DEFINE_string(flag_name="data_dir",
                           default_value="./CBTest/tang/",
                           docstring="CBT数据集的路径")

tf.app.flags.DEFINE_string(flag_name="output_dir",
                           default_value="",
                           docstring="临时目录")

tf.app.flags.DEFINE_string(flag_name="train_file",
                           default_value="train.txt",
                           docstring="CBT的训练文件")

tf.app.flags.DEFINE_string(flag_name="valid_file",
                           default_value="dev.txt",
                           docstring="CBT的验证文件")

tf.app.flags.DEFINE_string(flag_name="test_file",
                           default_value="test.txt",
                           docstring="CBT的测试文件")

tf.app.flags.DEFINE_string(flag_name="embedding_file",
                           default_value="./glove.6B.100d.sample.txt",
                           docstring="glove预训练的词向量文件")

tf.app.flags.DEFINE_integer(flag_name="max_vocab_num",
                            default_value=100000,
                            docstring="词库中存储的单词最大个数")

tf.app.flags.DEFINE_integer(flag_name="d_len_min",
                            default_value=0,
                            docstring="载入样本中文档的最小长度")

tf.app.flags.DEFINE_integer(flag_name="d_len_max",
                            default_value=2000,
                            docstring="载入样本中文档的最大长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_min",
                            default_value=0,
                            docstring="载入样本中问题的最小长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_max",
                            default_value=200,
                            docstring="载入样本中问题的最大长度")

# 模型超参数
tf.app.flags.DEFINE_integer(flag_name="hidden_size",
                            default_value=128,
                            docstring="RNN隐层数量")

tf.app.flags.DEFINE_integer(flag_name="num_layers",
                            default_value=1,
                            docstring="RNN层数")

tf.app.flags.DEFINE_bool(flag_name="use_lstm",
                         default_value="False",
                         docstring="RNN类型：LSTM或者GRU")

# 模型训练超参数
tf.app.flags.DEFINE_integer(flag_name="embedding_dim",
                            default_value=100,
                            docstring="词向量维度")

tf.app.flags.DEFINE_integer(flag_name="batch_size",
                            default_value=32,
                            docstring="batch_size")

tf.app.flags.DEFINE_integer(flag_name="num_epoches",
                            default_value=100,
                            docstring="epoch次数")

tf.app.flags.DEFINE_float(flag_name="dropout_rate",
                          default_value=0.2,
                          docstring="dropout比率")

tf.app.flags.DEFINE_string(flag_name="optimizer",
                           default_value="ADAM",
                           docstring="优化算法：SGD或者ADAM或者RMSprop")

tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=0.001,
                          docstring="SGD的学习率")

tf.app.flags.DEFINE_integer(flag_name="grad_clipping",
                            default_value=10,
                            docstring="梯度截断的阈值，防止RNN梯度爆炸")

FLAGS = tf.app.flags.FLAGS
# bucket，用来处理序列长度方差过大问题
d_bucket = ([150, 310], [310, 400], [450, 600], [600, 750], [750, 950])
q_bucket = (20, 40)

def get_cpu_frequency():
    cpu_frequency = {}
    try:
        cpu_frequency_all = psutil.cpu_freq()
        cpu_frequency['current'] = cpu_frequency_all.current
        cpu_frequency['min'] = cpu_frequency_all.min
        cpu_frequency['max'] = cpu_frequency_all.max
    except:
        return cpu_frequency

    return cpu_frequency


def get_cpu_time():
    cpu_time = {}
    cpu_time_all = psutil.cpu_times()
    cpu_time['user'] = cpu_time_all.user
    cpu_time['nice'] = cpu_time_all.nice
    cpu_time['system'] = cpu_time_all.system
    cpu_time['idle'] = cpu_time_all.idle
    cpu_time['iowait'] = cpu_time_all.iowait
    cpu_time['irq'] = cpu_time_all.irq
    cpu_time['softirq'] = cpu_time_all.softirq
    cpu_time['steal'] = cpu_time_all.steal
    cpu_time['guest'] = cpu_time_all.guest
    cpu_time['guest_nice'] = cpu_time_all.guest_nice

    return cpu_time


def get_all_disk_info():
    disk_all = []
    for num, device in enumerate(psutil.disk_partitions()):
        dic_disk_all = {}
        dic_disk_all['id'] = num
        dic_disk_all['device'] = device.device
        dic_disk_all['mountpoint'] = device.mountpoint
        dic_disk_all['fstype'] = device.fstype
        dic_disk_all['opts'] = device.opts
        dic_disk_all['maxfile'] = device.maxfile
        dic_disk_all['maxpath'] = device.maxpath
        disk_all.append(dic_disk_all)

    return disk_all


def get_disk_used():
    disk_used = {}
    disk_used_all = psutil.disk_usage('/')
    disk_used['total'] = disk_used_all.total
    disk_used['used'] = disk_used_all.used
    disk_used['free'] = disk_used_all.free
    disk_used['percent'] = disk_used_all.percent

    return disk_used


def get_disk_io():
    disk_io = {}
    disk_io_all = psutil.disk_io_counters()
    disk_io['read_count'] = disk_io_all.read_count
    disk_io['write_count'] = disk_io_all.write_count
    disk_io['read_bytes'] = disk_io_all.read_bytes
    disk_io['write_bytes'] = disk_io_all.write_bytes
    disk_io['read_time'] = disk_io_all.read_time
    disk_io['write_time'] = disk_io_all.write_time
    disk_io['read_merged_count'] = disk_io_all.read_merged_count
    disk_io['write_merged_count'] = disk_io_all.write_merged_count
    disk_io['busy_time'] = disk_io_all.busy_time

    return disk_io


def get_storage_info():
    storage = {}
    storage_all = psutil.swap_memory()
    storage['total'] = storage_all.total
    storage['used'] = storage_all.used
    storage['free'] = storage_all.free
    storage['percent'] = storage_all.percent
    storage['sin'] = storage_all.sin
    storage['sout'] = storage_all.sout

    return storage


def get_process_top3():
    process_top3 = []
    config = ['pid', 'name', 'status', 'create_time', 'memory_percent', 'io_counters', 'num_threads']
    for i, device in enumerate(psutil.process_iter(config)):
        process = {}

        device = device.info
        io_counters = device['io_counters']
        dic_io = {}
        if io_counters is not None:
            dic_io['read_count'] = io_counters.read_count
            dic_io['write_count'] = io_counters.write_count
            dic_io['read_bytes'] = io_counters.read_bytes
            dic_io['write_bytes'] = io_counters.write_bytes
            dic_io['read_chars'] = io_counters.read_chars
            dic_io['write_chars'] = io_counters.write_chars

        process['id'] = i
        process['pid'] = device['pid']
        process['name'] = device['name']
        process['status'] = device['status']
        process['create_time'] = device['create_time']
        process['memory_percent'] = round(device['memory_percent'], 2)  # 进程内存利用率
        process['io_counters'] = dic_io  # 进程的IO信息,包括读写IO数字及参数
        process['num_threads'] = device['num_threads']   # 进程开启的线程数
        process_top3.append(process)
        if i > 2:
            break

    return process_top3


def get_net_info():
    net = {}
    net_all = psutil.net_io_counters()
    net['bytes_sent'] = net_all.bytes_sent
    net['bytes_recv'] = net_all.bytes_recv
    net['packets_sent'] = net_all.packets_sent
    net['packets_recv'] = net_all.packets_recv
    net['errin'] = net_all.errin
    net['errout'] = net_all.errout
    net['dropin'] = net_all.dropin
    net['dropout'] = net_all.dropout

    return net


def read_byte(result_evaluate):
    # 计算磁盘写入字节数
    def is_Chinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    result_evaluate = json.dumps(result_evaluate, indent=4, ensure_ascii=False)
    byte_num = 0
    for i in result_evaluate:
        if is_Chinese(i):
                byte_num += 3
        else:
                byte_num += 1

    return byte_num


def CPU_Memory_utilization():
    # global cpu_utilization_list, memory_utilization_list
    # cpu_utilization_list = []
    # memory_utilization_list = []

    while True:
        cul_lis = psutil.cpu_percent(interval=1, percpu=True)
        cul = sum(cul_lis) / len(cul_lis)
        # cpu_utilization_list.append(cul)
        # memory_utilization_list.append(psutil.virtual_memory().percent)

        logging.info("CPU占用率:" + str(cul))
        storage = get_storage_info()
        storage['内存占用率'] = psutil.virtual_memory().percent
        logging.info("内存信息:" + str(storage))

        time.sleep(10)


def train_and_test(input_path, testfile):
    # 准备数据
    vocab_file, idx_train_file, idx_valid_file, idx_test_file = data_utils.prepare_cbt_data(
        input_path, FLAGS.train_file, FLAGS.valid_file, testfile, FLAGS.max_vocab_num, FLAGS.train,
        output_dir=FLAGS.output_dir)

    # 读取数据
    d_len_range = (FLAGS.d_len_min, FLAGS.d_len_max)
    q_len_range = (FLAGS.q_len_min, FLAGS.q_len_max)
    t_documents, t_questions, t_answer, t_candidates, t_ans_idx = data_utils.read_cbt_data(idx_train_file,
                                                                                d_len_range,
                                                                                q_len_range,
                                                                                max_count=FLAGS.max_count)
    v_documents, v_questions, v_answers, v_candidates, v_ans_idx = data_utils.read_cbt_data(idx_valid_file,
                                                                                 d_len_range,
                                                                                 q_len_range,
                                                                                 max_count=FLAGS.max_count)
    if FLAGS.test:
        test_documents, test_questions, test_answers, test_candidates, test_ans_idx = data_utils.read_cbt_data(
            idx_test_file,max_count=FLAGS.max_count)

    if FLAGS.predict:
        pred_documents, pred_questions, pred_answers, pred_candidates, pred_ans_idx = data_utils.read_cbt_data(
            idx_test_file, max_count=FLAGS.max_count)

    d_len = data_utils.get_max_length(t_documents)
    q_len = data_utils.get_max_length(t_questions)

    logging.info("-" * 50)
    logging.info("Building model with {} layers of {} units.".format(FLAGS.num_layers, FLAGS.hidden_size))

    # 初始化词向量矩阵，使用(-0.1,0.1)区间内的随机均匀分布
    word_dict = data_utils.load_vocab(vocab_file)
    embedding_matrix = data_utils.gen_embeddings(word_dict,
                                                 FLAGS.embedding_dim,
                                                 FLAGS.embedding_file,
                                                 init=np.random.uniform)

    if FLAGS.framework == "keras":
        # 使用keras版本的模型
        model = AttentionSumReader(word_dict, embedding_matrix, d_len, q_len,
                                   FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers,
                                   FLAGS.weight_path, FLAGS.use_lstm)
    else:
        # 使用tensorflow版本的模型
        sess = tf.Session()
        model = AttentionSumReaderTf(word_dict, embedding_matrix, d_len, q_len, sess,
                                     FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers,
                                     FLAGS.weight_path, FLAGS.use_lstm)

    if FLAGS.train:
        logging.info("Start training.")
        return model.train(train_data=(t_documents, t_questions, t_answer, t_candidates, t_ans_idx),
                    valid_data=(v_documents, v_questions, v_answers, v_candidates, v_ans_idx),
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.num_epoches_new,
                    opt_name=FLAGS.optimizer,
                    lr=FLAGS.learning_rate,
                    grad_clip=FLAGS.grad_clipping)

    if FLAGS.test:
        logging.info("Start testing.\nTesting in {} samples.".format(len(test_answers)))
        model.load_weight()
        return model.test(test_data=(test_documents, test_questions, test_answers, test_candidates, test_ans_idx),
                   batch_size=FLAGS.batch_size)

    if FLAGS.predict:
        logging.info("Start predicting.\nPredictint in {} samples.".format(len(pred_answers)))
        model.load_weight()
        return model.predict(predict_data=(pred_documents, pred_questions, pred_answers, pred_candidates, pred_ans_idx),
                          batch_size=FLAGS.batch_size)

    if FLAGS.ensemble:
        logging.info("Start ensemble testing.\nTesting in {} samples.".format(len(test_answers)))
        models = get_ensemble_model(word_dict, embedding_matrix, FLAGS.hidden_size, FLAGS.num_layers, FLAGS.use_lstm)
        return ensemble_test((test_documents, test_questions, test_answers, test_candidates), models)


def ensemble_test(test_data, models):
    data = [[] for _ in d_bucket]
    for test_document, test_question, test_answer, test_candidate in zip(*test_data):
        if len(test_document) <= d_bucket[0][0]:
            data[0].append((test_document, test_question, test_answer, test_candidate))
            continue
        if len(test_document) >= d_bucket[-1][-1]:
            data[len(models) - 1].append((test_document, test_question, test_answer, test_candidate))
            continue
        for bucket_id, (d_min, d_max) in enumerate(d_bucket):
            if d_min < len(test_document) < d_max:
                data[bucket_id].append((test_document, test_question, test_answer, test_candidate))
                continue

    acc, num = 0, 0
    for i in range(len(models)):
        num += len(data[i])
        logging.info("Start testing.\nTesting in {} samples.".format(len(data[i])))
        acc_i, _ = models[i].test(zip(*data[i]), batch_size=1)
        acc += acc_i
    logging.critical("Ensemble test done.\nAccuracy is {}".format(acc / num))


def get_ensemble_model(word_dict,
                       embedding_matrix,
                       hidden_size,
                       num_layers,
                       use_lstm):
    embedding_dim = len(embedding_matrix[0])
    models = {}
    for b_id, r in enumerate(d_bucket):
        weight_path = "{}{}-{}-{}-{}/".format(FLAGS.weight_path, r[0], r[1], q_bucket[0], q_bucket[1])
        model = AttentionSumReader(word_dict, embedding_matrix, r[1], q_bucket[1],
                                   embedding_dim, hidden_size, num_layers,
                                   weight_path, use_lstm)
        logging.info(weight_path)
        model.load_weight(weight_path)
        models[b_id] = model
    return models


def clear():
    """
    清除所有临时文件，请谨慎使用
    :return:
    """
    tmp_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir)
    if tf.gfile.Exists(tmp_dir):
        tf.gfile.DeleteRecursively(tmp_dir)

def save_arguments(args, file):
    with open(file, "w") as fp:
        json.dump(args, fp, sort_keys=True, indent=4)


def main(input_path, result_path):
    if not os.path.exists(FLAGS.weight_path):
        os.mkdir(FLAGS.weight_path)
    # 设置随机数种子
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    # 设置Log
    logging.basicConfig(filename=FLAGS.log_file,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%y-%m-%d %H:%M')
    #tf.app.run()

    if FLAGS.train:
        train_and_test(input_path)
    if FLAGS.predict:
        # transform file from json to train format
        random_ans_list = []
        with open(input_path + "/predict.txt", 'w') as wf:
            with open(input_path + "/RawData/input.json", 'r') as f:
                json_obj = json.load(f)
                if json_obj.get("data") is None or not hasattr(json_obj.get("data"), '__iter__'):
                    sys.exit()
                for element in json_obj.get("data"):
                    if element.get("paragraphs") is None or not hasattr(element.get("paragraphs"), '__iter__'):
                        continue
                    for para in element.get("paragraphs"):
                        if para.get("context") is None:
                            continue
                        if para.get("qas") is None or not hasattr(para.get("qas"), '__iter__'):
                            continue
                        wf.write(para.get("context") + "\n")
                        context_words_list = [word for word in jieba.cut(para.get("context"))]
                        for qas in para.get("qas"):
                            question = ""
                            if qas.get("question") is None:
                                continue
                            question += qas.get("question") + "XXXXX\t"
                            # a simple strategy for recall
                            question_words_list = [word for word in jieba.cut(qas.get("question"))]
                            queston_words_count_list = [para.get("context").count(word) for word in question_words_list]
                            queston_words_count_list = sorted(enumerate(queston_words_count_list), key=lambda x: x[
                                1], reverse=True)
                            candidate_list = []
                            for i in range(min(6, len(question_words_list))):
                                candidate_list.append(question_words_list[queston_words_count_list[i][0]])
                            if len(candidate_list) < 6:
                                for i in range(6 - len(candidate_list)):
                                    candidate_list.append(candidate_list[0])
                            random_ans_list.append(candidate_list)
                            question += candidate_list[0] + "\t" + "|".join(candidate_list)
                            wf.write(question + "\n")
        dict = train_and_test(input_path, "predict.txt")
        idx = 0
        for element in json_obj.get("data"):
            for para in element.get("paragraphs"):
                context = para.get("context")
                for qas in para.get("qas"):
                    if qas.get("answers") is None:
                        qas["answers"] = []
                    context_id = context.find(random_ans_list[idx][dict["predict"][idx]])
                    if context_id == -1:
                        idx += 1
                        continue
                    predict_text_length = random.randint(1,10)
                    if context_id + predict_text_length > len(context):
                        predict_text_length = len(context) - 1
                    else:
                        predict_text_length += context_id
                    if len(qas["answers"]) == 0:
                        qas["answers"].append({})
                    qas["answers"][0]["text"] = context[context_id : predict_text_length]
                    qas["answers"][0]["answer_start"] = context_id
                    idx += 1
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + "/result.json", "w", encoding='utf-8') as jsonf:
            json.dump(json_obj, jsonf, indent=4, ensure_ascii=False)
        return len(dict), dict["time"], json_obj
    if FLAGS.test:
        log_path = ""
        read_flie_size = 0
        for root, dirs, files in os.walk(input_path, topdown=True):
            for file in files:
                txtfile = os.path.join(root, file)
                log_path = os.path.join(root.replace(input_path, result_path), "log")
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                log_path = os.path.join(log_path, "info.log")
                read_flie_size += os.path.getsize(txtfile)

        for file in Path(input_path).glob('**/*.txt'):
            logging.info("file{}".format(file))
            mid = os.path.relpath(str(file), input_path)
            logging.info("mid{}".format(mid))
            dst_json = os.path.join(result_path, os.path.dirname(mid), str(file.stem) + '.json')
            logging.info("dst_json{}".format(dst_json))
            os.makedirs(os.path.dirname(dst_json), exist_ok=True)

            lis = []
            with open(input_path + "/Answer/input.json", 'r') as f:
                d = {}
                context = ""
                for line in f:
                    if isinstance(line, str):
                        elements = line.split("\t")
                        size = len(elements)
                    else:
                        elements = line.decode('utf-8').split("\t")
                        size = len(elements)
                    if size != 3:
                        d = {}
                        d["context"] = line
                        context = line
                    else:
                        d = {}
                        d["context"] = context
                        d["question"] = elements[0].replace("XXXXX", "")
                        d["predict"] = elements[1]
                        d["answer"] = [ans for ans in elements[2].split("|")]
                        lis.append(d)

            # 是否启动资源评测工具进行监控，默认不启动，启动时传参mon=yes(不为no都可以)
            if args.mon == 'no':
                dict = train_and_test(input_path, "/Answer/input.json")
                test_time = dict["time"]
                len_data = len(dict["label"])
                for idx, label in enumerate(dict["label"]):
                    lis[idx]["answer"] = lis[idx]["answer"][label]
                with open(dst_json, 'w', encoding='utf-8') as jsonf:
                    json.dump(lis, jsonf, indent=4, ensure_ascii=False)
            else:
                log = logging.getLogger()
                log.setLevel("INFO")
                file = logging.FileHandler(log_path, encoding="utf-8")
                fmt = '%(asctime)s--%(levelname)s-%(filename)s-%(lineno)d >>> %(message)s'
                pycharm_fmt = logging.Formatter(fmt=fmt)
                file.setFormatter(fmt=pycharm_fmt)
                log.addHandler(file)

                th = threading.Thread(target=CPU_Memory_utilization)
                th.setDaemon(True)  # 将子线程设置为守护线程（当主线程执行完成后，子线程无论是否执行完成都停止执行）
                th.start()

                time.sleep(1)
                dict = train_and_test(input_path, "/Answer/input.json")
                test_time = dict["time"]
                len_data = len(dict["label"])
                for idx in range(len(dict["label"])):
                    lis[idx]["answer"] = lis[idx]["answer"][dict["label"][idx]]
                with open(dst_json, 'w', encoding='utf-8') as jsonf:
                    json.dump(lis, jsonf, indent=4, ensure_ascii=False)
                write_flie_size = read_byte(lis)

                # 吞吐量及响应时间
                handling_capacity = {"handling_capacity(t/s)": round(len_data / test_time, 2),
                                     "numerator_test_len": len_data,
                                     "denominator_predict_time(s)": round(test_time, 2)}
                response_time = {"response_time(ms/t)": round(test_time / len_data * 1000, 2),
                                 "numerator_predict_time(s)": round(test_time, 2),
                                 "denominator_test_len": len_data}

                cpu_ = {"逻辑CPU个数": psutil.cpu_count(),
                        "物理CPU个数": psutil.cpu_count(logical=False),
                        "CPU频率": get_cpu_frequency(),
                        "CPU时间花费": get_cpu_time()}

                disk_ = {"程序读取字节数": read_flie_size,
                         "程序写入字节数": write_flie_size,
                         "所有已挂磁盘": get_all_disk_info(),
                         "磁盘使用情况": get_disk_used(),
                         "磁盘io统计": get_disk_io()}

                course_ = {"运行中进程top3": get_process_top3(),
                           "运行中全部进程": psutil.pids(),
                           "开机时间": psutil.boot_time()}

                logging.info("吞吐量:" + str(handling_capacity))
                logging.info("响应时间:" + str(response_time))
                logging.info("模型io字节及磁盘信息:" + str(disk_))
                logging.info("CPU信息:" + str(cpu_))
                logging.info("网卡信息:" + str(get_net_info()))
                logging.info("进程信息:" + str(course_))