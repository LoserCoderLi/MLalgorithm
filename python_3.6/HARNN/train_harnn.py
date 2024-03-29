# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging

sys.path.append('../../nlp_0/')
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
from text_harnn import TextHARNN
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

args = parser.parameter_parser()
OPTION = dh._option(pattern=0)
logger = dh.logger_fn("tflog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))

'''
将多个相关的数据序列打包在一起，形成一个对应的元组的序列。
这样做的好处是，可以在迭代过程中同时访问到对应的各个数据，
非常适合用于模型的训练过程中，其中需要同时处理多个相关的输入数据和标签。
'''
def create_input_data(data: dict):
    return zip(data['pad_seqs'], data['section'], data['subsection'],
               data['group'], data['subgroup'], data['onehot_labels'])


'''
主要作用是训练一个名为 HARNN 的模型

'''
def train_harnn():
    """Training HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)    ## 打印模型参数

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file) ## 加载 word2vec 模型,得到矩阵

    # Load sentences, labels, and training parameters,加载数据
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_data = dh.load_data_and_labels(args, args.train_file, word2idx)
    val_data = dh.load_data_and_labels(args, args.validation_file, word2idx)

    # Build a graph and harnn object
    '''
    使用 TensorFlow 构建计算图，并创建一个 HARNN 模型对象。
    这个模型的参数包括序列长度、词汇表大小、嵌入类型、嵌入大小、
    LSTM 隐藏层大小、注意力单元大小、全连接层隐藏层大小、类别数量等。
    '''
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            harnn = TextHARNN(
                sequence_length=args.pad_seq_len,
                vocab_size=len(word2idx),
                embedding_type=args.embedding_type,
                embedding_size=args.embedding_dim,
                lstm_hidden_size=args.lstm_dim,
                attention_unit_size=args.attention_dim,
                fc_hidden_size=args.fc_dim,
                num_classes_list=args.num_classes_list,
                total_classes=args.total_classes,
                l2_reg_lambda=args.l2_lambda,
                pretrained_embedding=embedding_matrix)

            # Define training procedure
            '''
            定义训练过程：定义学习率（使用指数衰减），优化器（使用 Adam），
            以及梯度剪裁。然后，定义训练操作，这个操作将梯度应用到变量上，并更新全局步骤。
            '''
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=harnn.global_step,
                                                           decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate,
                                                           staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(harnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=harnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            '''
            跟踪梯度：可选地，跟踪梯度的值和稀疏度，并将这些信息添加到摘要中。
            '''
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            '''
            准备输出目录：创建用于保存模型和摘要的目录。
            '''
            out_dir = dh.get_out_dir(OPTION, logger)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            '''
            创建摘要：创建用于记录训练损失和验证损失的摘要，并为这些摘要创建摘要写入器。
            '''
            loss_summary = tf.summary.scalar("loss", harnn.loss)

            # Train summaries
            '''
            创建模型保存器：创建一个 TensorFlow 保存器，用于保存模型的参数。
            '''
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if OPTION == 'R':
                # Load harnn model
                '''
                加载或初始化模型：如果在训练模式下，初始化所有的变量；如果在恢复模式下，从最新的检查点恢复模型。???????
                '''
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            if OPTION == 'T':
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(harnn.global_step)

            '''
            执行一个训练步骤，并更新模型的参数。
            batch_data:批量的训练数据
            feed_dict:这个字典将模型的输入和输出与实际的数据关联起来,也有其他参数如：harnn.is_training: True是否在训练
            
            调用 TensorFlow 的 sess.run 函数，执行一次训练操作 train_op。
            这个操作会利用当前的输入和输出数据，根据模型的结构和优化器，更新模型的参数。
            这个函数也返回了当前的步数、训练的摘要和损失。
            
            将训练的摘要写入到摘要写入器 train_summary_writer，并打印出当前的步数和损失。这样，我们就可以在训练过程中了解模型的表现。
            
            执行模型的一个训练步骤，更新模型的参数，并记录训练的信息。
            '''
            def train_step(batch_data):
                """A single training step."""
                x, sec, subsec, group, subgroup, y_onehot = zip(*batch_data)  ## 将这些数据解包并分配给不同的变量。

                feed_dict = {
                    harnn.input_x: x,
                    harnn.input_y_first: sec,
                    harnn.input_y_second: subsec,
                    harnn.input_y_third: group,
                    harnn.input_y_fourth: subgroup,
                    harnn.input_y: y_onehot,
                    harnn.dropout_keep_prob: args.dropout_rate,
                    harnn.alpha: args.alpha,
                    harnn.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, harnn.global_step, train_summary_op, harnn.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            '''
            函数接受一个参数 val_loader，这是一个加载验证数据的加载器。然后，它创建一个批量的验证数据。
            
            先初始化了一些变量，包括计数器、损失、精度、召回率和 F1 分数。它还创建了一些列表来存储真实的标签和预测的得分和标签。
            
            对于每一个验证数据批次，函数将数据解包并分配给不同的变量。
            然后，它创建一个字典 feed_dict，将模型的输入和输出与实际的数据关联起来。
            注意，这里的 dropout 概率被设置为 1.0，表示在验证过程中不使用 dropout。
            
            接着，它调用 TensorFlow 的 sess.run 函数，获取模型在当前批次数据上的得分和损失。
            然后，它将真实的标签和预测的得分添加到之前创建的列表中。
            
            然后，它使用阈值和 topK 方法来获取预测的标签，并将这些标签添加到相应的列表中。同时，它更新总的损失和计数器。
            
            在处理完所有的批次数据后，函数计算平均损失，
            然后计算各种评估指标，包括精度、召回率、F1 分数、AUC 和平均精度。这些指标用于评估模型在验证集上的性能。
            
            主要作用是在验证集上评估模型的性能，并返回各种评估指标。
            '''

            def validation_step(val_loader, writer=None):
                """Evaluates model on a validation set."""
                batches_validation = dh.batch_iter(list(create_input_data(val_loader)), args.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0
                eval_pre_tk = [0.0] * args.topK
                eval_rec_tk = [0.0] * args.topK
                eval_F1_tk = [0.0] * args.topK

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(args.topK)]

                for batch_validation in batches_validation:
                    x, sec, subsec, group, subgroup, y_onehot = zip(*batch_validation)
                    feed_dict = {
                        harnn.input_x: x,
                        harnn.input_y_first: sec,
                        harnn.input_y_second: subsec,
                        harnn.input_y_third: group,
                        harnn.input_y_fourth: subgroup,
                        harnn.input_y: y_onehot,
                        harnn.dropout_keep_prob: 1.0,
                        harnn.alpha: args.alpha,
                        harnn.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [harnn.global_step, validation_summary_op, harnn.scores, harnn.loss], feed_dict)

                    # Prepare for calculating metrics
                    for i in y_onehot:
                        true_onehot_labels.append(i)
                    for j in scores:
                        predicted_onehot_scores.append(j)

                    # Predict by threshold
                    batch_predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=scores, threshold=args.threshold)
                    for k in batch_predicted_onehot_labels_ts:
                        predicted_onehot_labels_ts.append(k)

                    # Predict by topK
                    for top_num in range(args.topK):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                        for i in batch_predicted_onehot_labels_tk:
                            predicted_onehot_labels_tk[top_num].append(i)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1
                eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_F1_ts = f1_score(y_true=np.array(true_onehot_labels),
                                      y_pred=np.array(predicted_onehot_labels_ts), average='micro')

                for top_num in range(args.topK):
                    eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                           y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                        y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F1_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                                   y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                   average='micro')

                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                         y_score=np.array(predicted_onehot_scores), average='micro')
                # Calculate the average PR
                eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                   y_score=np.array(predicted_onehot_scores), average='micro')

                return eval_loss, eval_auc, eval_prc, eval_pre_ts, eval_rec_ts, eval_F1_ts, \
                       eval_pre_tk, eval_rec_tk, eval_F1_tk

            # Generate batches
            batches_train = dh.batch_iter(list(create_input_data(train_data)), args.batch_size, args.epochs)
            num_batches_per_epoch = int((len(train_data['pad_seqs']) - 1) / args.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                train_step(batch_train)
                current_step = tf.train.global_step(sess, harnn.global_step)

                if current_step % args.evaluate_steps == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, \
                    eval_pre_ts, eval_rec_ts, eval_F1_ts, eval_pre_tk, eval_rec_tk, eval_F1_tk = \
                        validation_step(val_data, writer=validation_summary_writer)
                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))
                    # Predict by threshold
                    logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F1_ts))
                    # Predict by topK
                    logger.info("Predict by topK:")
                    for top_num in range(args.topK):
                        logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F1_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % args.checkpoint_steps == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))

    logger.info("All Done.")


if __name__ == '__main__':
    train_harnn()