import argparse
import csv
import time

import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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
    model = load_model(args.model)

    gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(args.input, args.gt)

    [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
    [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:args.length, :, :], x_acc[0:args.length, :, :]],
                                                 batch_size=1, verbose=1)

    gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

    gt_trajectory = gt_trajectory[0:args.length, :]


# ------------------------------save data----------------------------------
    #
    # with open("Ground_Truth_saved.csv", 'w', newline='') as gtf:
    # with open("Ground_Truth_saved_{}.csv".format(int(time.time())), 'w', newline='') as gtf:
    #     writer = csv.writer(gtf)
    #     writer.writerow(np.append(["GT_x"], gt_trajectory[:, 0]))
    #     writer.writerow(np.append(["GT_y"], gt_trajectory[:, 1]))
    #     writer.writerow(np.append(["GT_z"], gt_trajectory[:, 2]))
    #
    # #with open("Prediction_saved.csv", 'w', newline='') as gtf:
    # with open("Prediction_saved_{}.csv".format(int(time.time())), 'w', newline='') as pf:
    #     writer = csv.writer(pf)
    #     writer.writerow(np.append(["P_x"], pred_trajectory[:, 0]))
    #     writer.writerow(np.append(["P_y"], pred_trajectory[:, 1]))
    #     writer.writerow(np.append(["P_z"], pred_trajectory[:, 2]))
    modelname = args.model
    with open("GT_and_P_saved_{}.csv".format(int(time.time())), 'w', newline='') as mixed:
        writer = csv.writer(mixed)
        writer.writerow(np.array([modelname, args.length, args.data]))
        writer.writerow(np.append(["GT_x"], gt_trajectory[:, 0]))
        writer.writerow(np.append(["GT_y"], gt_trajectory[:, 1]))
        writer.writerow(np.append(["GT_z"], gt_trajectory[:, 2]))
        writer.writerow(np.append(["P_x"], pred_trajectory[:, 0]))
        writer.writerow(np.append(["P_y"], pred_trajectory[:, 1]))
        writer.writerow(np.append(["P_z"], pred_trajectory[:, 2]))


    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=[14.4, 10.8])
    ax = fig.gca(projection='3d')
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
    min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
    min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
    max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
    max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
    max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
    range_x = np.absolute(max_x - min_x)
    range_y = np.absolute(max_y - min_y)
    range_z = np.absolute(max_z - min_z)
    max_range = np.maximum(np.maximum(range_x, range_y), range_z)
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    ax.legend(['ground truth', 'predicted'], loc='upper right')
    plt.title("Total Second: {}s".format(args.length * stride / fps))
    plt.show()

if __name__ == '__main__':
    main()

