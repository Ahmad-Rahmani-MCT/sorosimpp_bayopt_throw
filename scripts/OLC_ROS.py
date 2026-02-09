#!/usr/bin/env python3

import numpy as np 
import rospy
import pandas as pd
from std_msgs.msg import Float64MultiArray 
import os

# user inputs 
optimal_input_filename = "optimal_inputs.csv"
QUEUE_SIZE = 1
NODE_FREQUENCY = 10.0  # [Hz] # node publishing frequency 
SOROSIM_TAG = "/sorosimpp" # initial part of the topic name to publish the actuation messages
N_ACT = 3  # Number of actuators

# reading .csv file and converting to np array 
script_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()  
optimal_input_path = os.path.join(script_path,optimal_input_filename) 
df =pd.read_csv(optimal_input_path, header=0) 
input = df.to_numpy() 
optimal_u = input
print("optimal input shape: ", optimal_u.shape)

## Optimal Controller Class
class Optimal_Controller:
    def __init__(self, n_act : int, u):

        #Instance attributes
        self.n_act = n_act 
        self.counter = 0 
        self.u = u 
        self.t0 = rospy.get_time() 

        # ROS Publisher
        self.pub_obj = rospy.Publisher(SOROSIM_TAG + "/actuators", Float64MultiArray, queue_size=QUEUE_SIZE) 

        # Wait for the publisher to establish connection with the logger 
        # start_wait = rospy.get_time()
        # rospy.loginfo("Waiting for subscribers (sorosimpp & logger) to connect...")
        while self.pub_obj.get_num_connections() < 2:
            if rospy.is_shutdown():
                return 
            # Optional: Timeout warning just in case something is wrong with the launch
            # if rospy.get_time() - start_wait > 5.0:
            #    rospy.logwarn_throttle(2, f"Still waiting... Current connections: {self.pub_obj.get_num_connections()}/2")
            rospy.sleep(0.1)
        # rospy.loginfo("Subscriber connected. Starting control loop.") 

        # Optional: Add a small sleep to ensure the logger's TF callback is also ready
        # rospy.sleep(0.5)

        # Initialize ROS message
        self.init_actMsg()

        # Main loop timer
        self.timer_obj = rospy.Timer(rospy.Duration(1.0 / NODE_FREQUENCY), self.main_loop)


    #Initial Actuation Message
    def init_actMsg(self):

        self.act_msg = Float64MultiArray() 
        '''
        start_time = rospy.get_time()
        rate = rospy.Rate(3)  # 3 Hz publishing frequency  

        while rospy.get_time() - start_time < 2.0:  # running for 2 second  
            self.act_msg.data = [0.0] * self.n_act
            self.pub_obj.publish(self.act_msg)
            rate.sleep()

        self.t0 = rospy.get_time() 
        '''

    #Updating the Actuator Message
    def update_actMsg(self):

        if self.counter < len(self.u) : 
            self.act_msg.data = self.u[self.counter,:].astype(float).tolist()  
            self.counter += 1
        else:
            # Logic added to stop the script once data is finished
            # rospy.loginfo("All commands sent. Shutting down.")
            # rospy.signal_shutdown("Finished sending optimal_u")
            pass
            
    #Main Loop called by the timer every defined freq
    def main_loop(self, event):
        # Update and publish message
        self.update_actMsg()
        
        # Only publish if we haven't shut down in the update step
        if not rospy.is_shutdown():
            self.pub_obj.publish(self.act_msg)

### Main ###
def main():
    rospy.init_node("openloop_controller", anonymous=True)
    controller = Optimal_Controller(N_ACT, optimal_u) 
    rospy.spin()

# Execute
if __name__ == '__main__':
    main()