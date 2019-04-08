# -*- coding: UTF-8 -*-
from utils import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cat_list = [i for i in os.listdir("./dataset") if os.path.isdir(os.path.join("./dataset", i))]
NUM_PTS = 4096
if not os.path.exists("test_results"):
    os.mkdir("test_results")
test_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))

test_dir = os.path.join("test_results", test_time)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for cat in cat_list:

    cat_dir = os.path.join(test_dir, cat)

    cat_name = cat
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)

    flog = open(os.path.join(cat_dir, 'log_test.txt'), 'w')
    test_coordinates, test_ncoordinates, test_ocoordinates = load_single_cat_h5(cat, NUM_PTS, "test", "coordinates",
                                                                                              "ncoordinates",
                                                                                              "ocoordinates")

    nb_samples = test_coordinates.shape[0]
    modelPath = "./train_results/2019_04_03_17_01/{}/model/".format(cat)
    model_id = 198
    graph_file = os.path.join(modelPath, "model-" + str(model_id) + ".meta")
    variable_file = os.path.join(modelPath, "model-" + str(model_id))
    GAN_graph = tf.Graph()


    def log_string(out_str):
        flog.write(out_str+'\n')
        flog.flush()
        print(out_str)
    log_string(graph_file)

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(graph_file)
        saver.restore(sess, variable_file)

        fake_pts = tf.get_default_graph().get_tensor_by_name("generator_2/Tanh:0")
        input_pt = tf.get_default_graph().get_tensor_by_name("all_point_cloud:0")
        batch_size = int(input_pt.get_shape()[0])
        # print("batch_size:", batch_size)
        bn_is_train = tf.get_default_graph().get_tensor_by_name("bn_is_train:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")

        total_batch = test_coordinates.shape[0] // batch_size
        for i in range(total_batch):
            start_idx = batch_size * i
            end_idx = batch_size * (i+1)
            batch_test_coordinates = test_coordinates[start_idx:end_idx]
            # print(batch_test_coordinates.shape)
            batch_test_ncoordinates = test_ncoordinates[start_idx:end_idx]
            batch_test_ocoordinates = test_ocoordinates[start_idx:end_idx]
            batch_double_coordinates = np.concatenate([batch_test_coordinates, batch_test_ncoordinates], axis=2)
            new_coordiantes = sess.run(fake_pts, feed_dict={input_pt: batch_double_coordinates, bn_is_train: False})
            new_coordiantes = np.squeeze(new_coordiantes)
            # print("new.shape:",new_coordiantes.shape)
            for j in range(batch_size):
                # fname_GT = os.path.join(cat_dir, "test_chair_GT_{}.png".format(i * batch_size + j))
                # fname_input = os.path.join(cat_dir, "test_chair_input_{}.png".format(i * batch_size +j))
                # fname_gen = os.path.join(cat_dir, "test_chair_gen_{}.png".format(i * batch_size +j))
                fname_ = os.path.join(cat_dir, "test_{0}_{1}.png".format(cat_name, i * batch_size + j))
                # display_only(pts=batch_test_coordinates[j], name_=fname_GT)
                # display_only(pts=batch_test_ocoordinates[j], name_=fname_input)
                # display_only(pts=new_coordiantes[j], name_=fname_gen)
                display(batch_test_ocoordinates[j], new_coordiantes[j], name_=fname_)

                # fout = os.path.join(cat_dir, "test_{0}_{1}.png".format(cat_name, i * batch_size + j))

                # horizontal_concatnate_pic(fout, fname_input, fname_GT, fname_gen)

                # os.remove(fname_GT)
                # os.remove(fname_gen)
                # os.remove(fname_input)
            flog.close()
    print("完成{}".format(cat))




