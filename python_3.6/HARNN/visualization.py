# -*- coding:utf-8 -*-
__author__ = 'Randolph'
'''
主要作用是可视化 HARNN 模型的注意力权重
'''

import sys
import time
import logging

sys.path.append('../../nlp_0/')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


def create_input_data(data: dict):
    return zip(data['pad_seqs'], data['content'], data['section'], data['subsection'], data['group'],
               data['subgroup'], data['onehot_labels'])

'''
作用是对输入的 visual_list 进行归一化处理。
首先，它找到 visual_list 中前 visual_len 个元素的最小值 min_weight 和最大值 max_weight。
然后，计算出这些元素的范围 margin（即最大值减去最小值）
接着，对 visual_list 中的每一个元素，计算其归一化的值。这是通过将每个元素减去最小值，然后除以范围来实现的。为了防止除以零的情况，这里加上了一个很小的数 epsilon。
最后，将所有的归一化值收集到一个新的列表 result 中，并返回这个列表。
'''
def normalization(visual_list, visual_len, epsilon=1e-12):
    min_weight = min(visual_list[:visual_len])
    max_weight = max(visual_list[:visual_len])
    margin = max_weight - min_weight

    result = []
    for i in range(visual_len):
        value = (visual_list[i] - min_weight) / (margin + epsilon)
        result.append(value)
    return result

'''
作用是创建一个 HTML 文件来可视化模型的注意力权重。
首先，它打开一个名为 'attention.html' 的文件进行写入操作。
然后，它写入 HTML 的基本结构，包括 <html>, <body>, 和 <div> 标签。
对于 visual_list 中的每个 visual（即每个注意力权重列表），它创建一个新的 <p> 段落。
在每个段落中，对于输入序列 input_x 的每个词，它创建一个新的 <span> 元素。这个元素的背景颜色的红色通道的透明度由对应的注意力权重 alpha 决定，而元素的文本则是对应的词。(有点像热力图)
之后，它关闭当前的 <p> 段落，并在处理完 visual_list 中的所有注意力权重列表后，关闭 <div>，<body> 和 <html> 标签。
最后，它关闭文件.
'''
def create_visual_file(input_x, visual_list: list, seq_len):
    f = open('attention.html', 'w')
    f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')
    f.write('<div style="margin:25px;">\n')
    for visual in visual_list:
        f.write('<p style="margin:10px;">\n')
        for i in range(seq_len):
            alpha = "{:.2f}".format(visual[i])
            word = input_x[0][i]
            f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,{0})">{1}</span>\n'
                    .format(alpha, word))
        f.write('</p>\n')
    f.write('</div>\n')
    f.write('</body></html>')
    f.close()
'''
可视化 HARNN 模型的注意力权重。
'''
def visualize():
    """Visualize HARNN model."""

    # Load word2vec model
    '加载 word2vec 模型'
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    '加载数据'
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args, args.test_file, word2idx)

    # Load harnn model
    '加载 HARNN 模型'
    OPTION = dh._option(pattern=1)
    if OPTION == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    else:
        logger.info("Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(CPT_DIR)
    logger.info(checkpoint_file)

    graph = tf.Graph()  ## 创建 TensorFlow 会话：在新的图上创建会话，并配置会话参数。
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables 恢复模型：加载保存的元图并恢复变量。
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))##加载模型结构
            saver.restore(sess, checkpoint_file) ## 恢复参数

            # Get the placeholders from the graph by name
            '在 TensorFlow 中，占位符（Placeholder）是一种特殊的变量，用于接收输入数据。在定义模型的时候，我们通常会创建一些占位符，然后在运行模型的时候，通过 feed_dict 参数将数据填充到这些占位符中。'
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y_first = graph.get_operation_by_name("input_y_first").outputs[0]
            input_y_second = graph.get_operation_by_name("input_y_second").outputs[0]
            input_y_third = graph.get_operation_by_name("input_y_third").outputs[0]
            input_y_fourth = graph.get_operation_by_name("input_y_fourth").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            '定义和操作张量（Tensor）来构建计算图。张量是一个多维数组，所有的数据都通过张量在计算图中流动。在运行会话（Session）时，我们通常会选择一些张量来评估，这些张量通常代表了模型的输出或者中间层的结果。'

            first_visual = graph.get_operation_by_name("first-output/visual").outputs[0]
            second_visual = graph.get_operation_by_name("second-output/visual").outputs[0]
            third_visual = graph.get_operation_by_name("third-output/visual").outputs[0]
            fourth_visual = graph.get_operation_by_name("fourth-output/visual").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            '''有多个输出节点，你可能需要将这些节点的名称通过 '|' 分隔开来。然后，你可以通过 graph.get_tensor_by_name 函数来获取这些节点'''

            output_node_names = "first-output/visual|second-output/visual|third-output/visual|fourth-output/visual|output/scores"

            # Save the .pb model file
            '''将 TensorFlow 模型保存为 .pb 文件是一种常见的做法，因为 .pb 文件可以将模型的结构和参数一起保存
            '''
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            '生成一个时期（epoch）的批次（batches)'
            batches = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)

            for batch_test in batches:
                x, x_content, sec, subsec, group, subgroup, y_onehot = zip(*batch_test)

                feed_dict = {
                    input_x: x,
                    input_y_first: sec,
                    input_y_second: subsec,
                    input_y_third: group,
                    input_y_fourth: subgroup,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_first_visual, batch_second_visual, batch_third_visual, batch_fourth_visual = \
                    sess.run([first_visual, second_visual, third_visual, fourth_visual], feed_dict)

                batch_visual = [batch_first_visual, batch_second_visual, batch_third_visual, batch_fourth_visual]

                seq_len = len(x_content[0])
                pad_len = len(batch_first_visual[0])
                length = (pad_len if seq_len >= pad_len else seq_len)
                visual_list = []

                for visual in batch_visual:
                    visual_list.append(normalization(visual[0].tolist(), length))

                create_visual_file(x_content, visual_list, seq_len)
    logger.info("Done.")


if __name__ == '__main__':
    visualize()
