import rospy
import math
import time
import copy
import random
import torch
import numpy as np
from shapely.geometry import Point
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge, CvBridgeError
from typing import OrderedDict
import sys
sys.path.append('../../')
from training.utility import *


class RandEvalGpu:
    """ Perform Random Evaluation on GPU """
    def __init__(self,
                 actor_net,
                 robot_init_pose_list,
                 goal_pos_list,
                 obstacle_poly_list,
                 ros_rate=10,
                 max_steps=2000, 
                 min_spd=0.05,
                 max_spd=0.5,
                 is_spike=False,
                 is_scale=False,
                 is_poisson=False,
                 batch_window=50,
                 action_rand=0.05,
                 depth_img_dim=(48, 64),
                 goal_dis_min_dis=0.3,
                 goal_th=0.5,
                 obs_near_th=0.18,
                 use_cuda=True,
                 is_record=False):
        """
        :param actor_net: Actor Network
        :param robot_init_pose_list: robot init pose list
        :param goal_pos_list: goal position list
        :param obstacle_poly_list: obstacle list
        :param ros_rate: ros rate
        :param max_steps: max step for single goal
        :param min_spd: min wheel speed
        :param max_spd: max wheel speed
        :param is_spike: is using SNN
        :param is_scale: is scale DDPG state input
        :param is_poisson: is use rand DDPG state input
        :param batch_window: batch window of SNN
        :param action_rand: random of action
        :param scan_half_num: half number of scan points
        :param scan_min_dis: min distance of scan
        :param goal_dis_min_dis: min distance of goal distance
        :param goal_th: distance for reach goal
        :param obs_near_th: distance for obstacle collision
        :param use_cuda: if true use cuda
        :param is_record: if true record running data
        """
        self.actor_net = actor_net
        self.robot_init_pose_list = robot_init_pose_list
        self.goal_pos_list = goal_pos_list
        self.obstacle_poly_list = obstacle_poly_list
        self.ros_rate = ros_rate
        self.max_steps = max_steps
        self.min_spd = min_spd
        self.max_spd = max_spd
        self.is_spike = is_spike
        self.is_scale = is_scale
        self.is_poisson = is_poisson
        self.batch_window = batch_window
        self.action_rand = action_rand
        self.goal_dis_min_dis = goal_dis_min_dis
        self.goal_dis_scale = 1
        self.depth_scale = 1.0
        self.depth_min_dis = 0.05
        self.spike_state_num = 198
        self.cv_bridge = CvBridge()
        self.goal_th = goal_th
        self.obs_near_th = obs_near_th
        self.use_cuda = use_cuda
        self.is_record = is_record
        self.record_data = []
        # Put network to device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.actor_net.to(self.device)
        # Robot State
        self.depth_img_dim = depth_img_dim
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_pose = [0., 0., 0.]
        self.robot_spd = [0., 0.]
        self.robot_depth_img = np.zeros(self.depth_img_dim)
        self.robot_2_target_dis = 0
        self.robot_2_target_dir = 0
        self.linear_spd_max = 0.5
        self.linear_spd_min = 0.05
        # Subscriber
        rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('/kinect/depth/image_raw', Image, self._robot_depth_cb)
        rospy.Subscriber('/target_position', Target, self._robot_target_cb)
        # Publisher
        self.pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Service
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        # Init Subscriber
        while not self.robot_state_init:
            continue
        while not self.robot_scan_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

        # load trained CNN 
        self.trained_CNN = Trained_CNN()
        file_name = 'DDPG_R1_cnn_network_s8.pt'
        state_dict = torch.load('../save_ddpg_weights/' + file_name)
        new_state_dict = OrderedDict()
        new_state_dict = {'net1.'+k:v for k,v in state_dict.items()}
        net_cnn_dict = self.trained_CNN.state_dict()
        net_cnn_dict.update(new_state_dict)
        self.trained_CNN.load_state_dict(net_cnn_dict)
        self.trained_CNN.to(self.device)       


    def _wheeled_network_2_robot_action_decoder(action, wheel_max, wheel_min, diff=0.25):
        """
        Decode wheeled action from network to linear and angular speed for the robot
        :param action: action for wheel spd
        :param wheel_max: max wheel spd
        :param wheel_min: min wheel spd
        :param diff: diff of wheel for decoding angular spd
        :return: robot_action
        """
        l_spd = action[0] * (wheel_max - wheel_min) + wheel_min
        r_spd = action[1] * (wheel_max - wheel_min) + wheel_min
        linear = (l_spd + r_spd) / 2
        angular = (r_spd - l_spd) / diff
        return [linear, angular]


    def _ddpg_state_2_spike_value_state(self, state, normal_state_num,
                                    goal_dir_range=math.pi, linear_spd_range=0.5, angular_spd_range=2.0):
        """
        Transform DDPG state to Spike Value State for SNN
        :param state: single ddpg state
        :param spike_state_num: number of spike states
        :param goal_dir_range: range of goal dir
        :param linear_spd_range: range of linear spd
        :param angular_spd_range: range of angular spd
        :return: spike_state_value
        """
        out_state = []
        normal_state, depth_state = state
        tmp_normal_state = [0 for _ in range(normal_state_num)]
        if normal_state[0] > 0:
            tmp_normal_state[0] = normal_state[0] / goal_dir_range
            tmp_normal_state[1] = 0
        else:
            tmp_normal_state[0] = 0
            tmp_normal_state[1] = abs(normal_state[0]) / goal_dir_range
        tmp_normal_state[2] = normal_state[1]
        tmp_normal_state[3] = normal_state[2] / linear_spd_range
        if normal_state[3] > 0:
            tmp_normal_state[4] = normal_state[3] / angular_spd_range
            tmp_normal_state[5] = 0
        else:
            tmp_normal_state[4] = 0
            tmp_normal_state[5] = normal_state[3] /angular_spd_range
        
        out_state.append(tmp_normal_state)
        out_state.append(depth_state)

        return out_state

    def _robot_state_2_ddpg_state(self, state):   # [dis, dir, sqrt(linear.x**2 + linear.y**2), angular.z, depth]
        """
        Transform robot state to DDPG state
        ## Robot State: [robot_pose, robot_spd, scan]
        Robot State: [robot_pose, robot_spd, robot_depth]
        DDPG state: [Distance to goal, Direction to goal, Linear Spd, Angular Spd, depth_img]
        :param state: robot state
        :return: ddpg_state
        """
        tmp_goal_dis = state[0]  # goal_dis
        if tmp_goal_dis == 0:
            tmp_goal_dis = self.goal_dis_scale   # 1
        else:
            tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis  # 0.5 / tmp_goal_dis 
            if tmp_goal_dis > 1:
                tmp_goal_dis = 1
            tmp_goal_dis = tmp_goal_dis * self.goal_dis_scale    # tmp_goal_dis * 1
        ddpg_state = [[state[1], tmp_goal_dis, state[2], state[3]]]   # [Direction to goal, Distance to goal, sqrt(x**2 + y**2), msg.twist[-1].angular.z]
        '''
        Transform distance in laser scan to [0, scale]
        '''
        rescale_depth_img = self.depth_scale * (self.depth_min_dis / state[2])
        rescale_depth_img = np.clip(rescale_depth_img, 0, self.depth_scale)
        ddpg_state.append(rescale_depth_img)
        return ddpg_state   # [list(1x4), np.array(1x48x64)]


    def _state_2_state_spikes(self, spike_state_value, batch_size):   # 
        """
        Transform state to spikes of input neurons
        :param spike_state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: state_spikes
        """
        spike_state_value = spike_state_value.reshape((-1, self.spike_state_num, 1)) # np: [1x198x1]
        state_spikes = np.random.rand(batch_size, self.spike_state_num, self.batch_window) < spike_state_value
        state_spikes = state_spikes.astype(float)
        state_spikes = torch.Tensor(state_spikes).to(self.device)
        return state_spikes


    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        if self.robot_state_init is False:
            self.robot_state_init = True
        quat = [msg.pose[-1].orientation.x,
                msg.pose[-1].orientation.y,
                msg.pose[-1].orientation.z,
                msg.pose[-1].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
        self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
        self.robot_spd = [linear_spd, msg.twist[-1].angular.z]
        

    def _robot_depth_cb(self, msg):
        """
        Callback function for robot depth image
        :param msg: depth msg
        """
        if self.robot_depth_init is False:
            self.robot_depth_init = True
        tmp_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")     # from msg to cv_img (480 x 640)
        np.save("./tmp_data.npy", tmp_depth_img)
        #print("real data: ", tmp_depth_img)
        tmp_data = np.zeros((self.depth_img_dim))
        for i in range(0, 480, 10):
            for j in range(0, 640, 10):
                if tmp_depth_img[i][j] == 0:
                    tmp_data[int(i/10)][int(j/10)] = 25
                else:
                    tmp_data[int(i/10)][int(j/10)] = tmp_depth_img[i][j]
        self.robot_depth_img = tmp_data[np.newaxis, :]   # change into (1 x 480 x 640)

    
    def _robot_target_cb(self, msg):
        """
        Callback function for robot target msg
        :param msg: target position
        """
        self.robot_2_target_dis = msg.dis
        self.robot_2_target_dir = msg.dir

        tmp_goal_dis = copy.deepcopy(self.robot_2_target_dis)
        tmp_goal_dir = copy.deepcopy(self.robot_2_target_dir)
        tmp_robot_spd = copy.deepcopy(self.robot_spd)
        tmp_robot_depth = copy.deepcopy(self.robot_depth_img)

        ddpg_state = [tmp_goal_dis, tmp_goal_dir, tmp_robot_spd[0], tmp_robot_spd[1], tmp_robot_depth]  # [dir, dis, sqrt(linear.x**2 + linear.y**2), angular.z, depth]
        ddpg_state = self._robot_state_2_ddpg_state(ddpg_state)   # [list(1x4), np.array(1x48x64)] -> [Distance to goal, Direction to goal, Linear Spd, Angular Spd, depth_img]
        spike_state_value = self._ddpg_state_2_spike_value_state(ddpg_state, 6)   # state: [list(1x6), np.array(1x48x64)]

        #print('flag = ', flag)
        with torch.no_grad():
            normal_state, depth_state = spike_state_value
            depth_state = torch.Tensor(depth_state).unsqueeze(0).to(self.device)  # tensor(1x1x48x64)
            depth_out = self.trained_CNN(depth_state)   # tensor(1 x 192)
            depth_out = depth_out.cpu().numpy()
            normal_state = np.array([normal_state])
            combined_data = np.concatenate([normal_state, depth_out], axis=1)  # np: 1 x 198               
        
            #print('shape: ', combined_data.shape)
            state_spikes = self._state_2_state_spikes(combined_data, 1)
            raw_action = self.actor_net(state_spikes, 1).to(self.device)


        decode_action = self._wheeled_network_2_robot_action_decoder(
            raw_action, self.linear_spd_max, self.linear_spd_min
        )

        move_cmd = Twist()
        move_cmd.linear.x = decode_action[0]
        move_cmd.angular.z = decode_action[1]
        self.pub_action.publish(move_cmd)