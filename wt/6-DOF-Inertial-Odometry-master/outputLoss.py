import argparse
import csv
import time

import tensorflow
from tensorflow.keras.models import load_model
from model import CustomMultiLossLayer
from dataset import *
from util import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')', default='oxiod')
    parser.add_argument('--model', help='Model path', default="epo20_bs32_adam0.0001_lossnoadj_1657277198.hdf5")
    # '6dofio_oxiod.hdf5'
    parser.add_argument('--input', help='Input sequence path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv\" for OxIOD, \"MH_02_easy/mav0/imu0/data.csv\" for EuRoC)',
                        default="Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv")
    parser.add_argument('--gt', help='Ground truth path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv\" for OxIOD, \"MH_02_easy/mav0/state_groundtruth_estimate0/data.csv\" for EuRoC)',
                        default="Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv")
    parser.add_argument('--data',
                        help='GT & Input',
                        default="handheld")
    parser.add_argument('--length',
                        help='数据选取长度，默认200',
                        default=200)
    args = parser.parse_args()



# ---------------------------------get dataset--------------------------
    if(True):
        if args.data == "handheld":
            args.input = "Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv"
        elif args.data == "handbag":
            args.input = "Oxford Inertial Odometry Dataset/handbag/data1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/handbag/data1/syn/vi1.csv"
        elif args.data == "pocket":
            args.input = "Oxford Inertial Odometry Dataset/pocket/data2/syn/imu6.csv"
            args.gt = "Oxford Inertial Odometry Dataset/pocket/data2/syn/vi6.csv"
        elif args.data == "running":
            args.input = "Oxford Inertial Odometry Dataset/running/data1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/running/data1/syn/vi1.csv"
        elif args.data == "sw":
            args.input = "Oxford Inertial Odometry Dataset/slow walking/data1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/slow walking/data1/syn/vi1.csv"
        elif args.data == "LS":
            args.input = "Oxford Inertial Odometry Dataset/large scale/floor1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/large scale/floor1/syn/tango1.csv"
        elif args.data == "trolley":
            args.input = "Oxford Inertial Odometry Dataset/trolley/data1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/trolley/data1/syn/vi1.csv"
        elif args.data == "virtual":
            args.input = "Oxford Inertial Odometry Dataset/vertualdatastill/data1/syn/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/vertualdatastill/data1/syn/vi1.csv"
        elif args.data == "nrc01":
            args.input = "Oxford Inertial Odometry Dataset/nrc/data1/imu1.csv"
            args.gt = "Oxford Inertial Odometry Dataset/nrc/data1/vi1.csv"
        elif args.data == "nrc02":
            args.input = "Oxford Inertial Odometry Dataset/nrc/data1/imu2.csv"
            args.gt = "Oxford Inertial Odometry Dataset/nrc/data1/vi2.csv"
        elif args.data == "nrc03":
            args.input = "Oxford Inertial Odometry Dataset/nrc/data2/imu5.csv"
            args.gt = "Oxford Inertial Odometry Dataset/nrc/data2/vi5.csv"

    window_size = 200
    stride = 10
    fps = 100

    # from tensorflow.keras.models import load_model
    # model = load_model("6dofio_oxiod.hdf5")
    model = load_model("TRAINMODEL_epo1_bs128_adam0.0001_lossnoadj_1657288018.hdf5",
                            custom_objects={'CustomMultiLossLayer': CustomMultiLossLayer}, compile=False)

    gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(args.input, args.gt)
    [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)

    middle_model = tensorflow.keras.Model(inputs=[x_gyro[0:args.length], x_acc[0:args.length], y_delta_p[0:args.length], y_delta_q[0:args.length]],
                                          outputs=model.get_layer('custom_multi_loss_layer').output)
    middle_output = middle_model.predict([x_gyro[0:args.length], x_acc[0:args.length], y_delta_p[0:args.length], y_delta_q[0:args.length]]
                                         , batch_size=1, verbose=1)

    theLoss = model.predict([x_gyro[0:args.length], x_acc[0:args.length], y_delta_p[0:args.length], y_delta_q[0:args.length]],
                                                 batch_size=1, verbose=1)

    print(middle_output)
    print("Loss shape is:", len(theLoss))

    modelname = args.model
    with open("Losses_saved_{}.csv".format(int(time.time())), 'w', newline='') as mixed:
        writer = csv.writer(mixed)
        writer.writerow(np.array([modelname, args.length, args.data]))
        writer.writerow(np.append(["Loss"], theLoss))



if __name__ == '__main__':
    main()

