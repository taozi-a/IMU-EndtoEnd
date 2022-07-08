import argparse
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
#from sklearn.externals import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')', default="oxiod")
    parser.add_argument('--model', help='Model path', default="six_dof_train_model_1656942162.hdf5")
    args = parser.parse_args()

    model = load_model(args.model)

    window_size = 200
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu2.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu5.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu6.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu1.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu3.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu1.csv')

    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi2.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi5.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi6.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi1.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi3.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi1.csv')



    for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
        gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)

        gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        pred_trajectory = pred_trajectory[0:200, :]
        gt_trajectory = gt_trajectory[0:200, :]

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))

        print('Trajectory RMSE, sequence %s: %f' % (cur_imu_data_filename, trajectory_rmse))

if __name__ == '__main__':
    main()