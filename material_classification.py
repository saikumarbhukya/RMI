import copy
import cv2
import sys
import os
import json
import time
from datetime import datetime, timedelta
import traceback
import numpy as np
from collections import defaultdict
from collections import deque

def decode_img(hex_str):
    img_enc = bytes.fromhex(hex_str)
    jpg = np.frombuffer(img_enc, dtype=np.uint8)
    img = cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)
    return img


def encode_img(img):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encode = cv2.imencode('.jpg', img, encode_params)
    output = img_encode.tobytes().hex()
    return output

class MaterialDetection:
    def __init__(self):
        self.use_case = ''
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.key_string_prev_time_stamp = None  # R
        self.key_string_conta_time_stamp = None  # R
        self.key_string_prev_material_state = None  # R
        self.key_string_prev_event_state = "False"  # R
        self.key_string_freeze_contamination = "True"  # R
        self.key_string_start_10min_cycle = "True"  # R
        self.key_string_freezing_period = "False"  # R
        self.key_string_empty_belt_start_10min = "True"  # R
        self.key_string_contamination_20sec = "True"  # R
        self.time_tracking_10min_enabled = "False"  # R
        self.key_string_start_20sec_cycle = "True"  # R

        # Queue to keep track of the last detected classes for stabilization (history-based smoothing)
        self.class_history = deque(maxlen=10)  # Adjust the maxlen as needed for stabilization

    def get_stabilized_class(self):
        """
        Determines the most frequent class in the history queue for stabilization.
        Returns the most frequent class in the queue.
        """
        if not self.class_history:
            return None
        # Count the occurrences of each class in the history
        class_counts = {}
        for detected_class in self.class_history:
            if detected_class in class_counts:
                class_counts[detected_class] += 1
            else:
                class_counts[detected_class] = 1
        # Find the class with the maximum count (most stable class)
        stabilized_class = max(class_counts, key=class_counts.get)
        return stabilized_class

    def validate_detection(self, utility, message, rule_set=None):
        try:
            self.utility = utility
            self.use_case_name = message.get('current_usecase')
            mat_class, mat_conf, _ = message.get('inf_op')
            self.utility.loginfo(f"Current Material Classified: [{mat_class}]")
            '''
            self.key_string_current_material_state = '.'.join(
                [message.get('area'), self.use_case_name, 'Current_Material_State'])

            self.key_string_contamination_material_state = '.'.join(
                [message.get('area'), self.use_case_name, 'Contamination_Material'])
            '''
            self.key_string_prev_state_image = '.'.join([message.get('area'), self.use_case_name, 'Prev_State_Image'])
            self.key_string_contamination_state_image = '.'.join(
                [message.get('area'), self.use_case_name, 'Contamination_State_Image'])

            # If the confidence level is below 0.3, ignore this detection and return early
            if mat_conf < 0.3:
                return None

            # Add the detected class to the history queue for stabilization
            self.class_history.append(mat_class)
            # Get the stabilized material class based on history
            stabilized_class = self.get_stabilized_class()

            # If the stabilized class is not "Empty_Belt", update the current material state
            if stabilized_class != "Empty_Belt":
                # Changing Current state only if it is not Empty Belt
                self.key_string_current_material_state = stabilized_class
                #self.key_string_prev_state_mat_class = self.key_string_current_material_state
                self.utility.loginfo(f"Current Material State in Redis  {self.key_string_current_material_state}")
                if self.key_string_prev_material_state is None:  ##Previous state is None
                    ##Previous_State = Current_State
                    self.key_string_prev_material_state = self.key_string_current_material_state
                    self.key_string_prev_time_stamp = datetime.now().strftime(self.time_format)
                    self.utility.master_redis.set_val(key=f"{self.key_string_prev_state_image}",
                                                      val=utility.master_redis.get_val(message.get('inf_op_img_uid')))
                    self.utility.loginfo(f"Previous Material State in redis {self.key_string_prev_material_state}")
                    self.utility.loginfo(f"Previous Material Timestamp in redis {self.key_string_prev_time_stamp}")

            # Check if the stabilized class is "Empty_Belt" or if the 10-minute cycle has started
            if stabilized_class == "Empty_Belt" or self.time_tracking_10min_enabled == "True":
                # Once it is in Empty Belt, Start the 15min time
                if self.key_string_start_10min_cycle == "True":
                    ## Initiating the time tracker
                    time_start_10min = datetime.now().strftime(self.time_format)
                    self.key_string_empty_belt_start_10min = time_start_10min  # change prev tp empty_belt_start
                    self.key_string_start_10min_cycle = "False"
                    self.time_tracking_10min_enabled = "True"

                time_now_10min = (datetime.strptime((str(datetime.now())).split('.')[0], self.time_format))
                time_diff_10 = (time_now_10min - datetime.strptime(self.key_string_empty_belt_start_10min,
                                                                   self.time_format)).total_seconds()

                if time_diff_10 < 601:
                    self.utility.loginfo(
                        f"Printing self.key_string_current_material_state {self.key_string_current_material_state}")
                    if self.key_string_current_material_state != self.key_string_prev_material_state:  # and self.time_tracking_enabled == "False":
                        if self.key_string_start_20sec_cycle == "True":
                            time_start_20sec = datetime.now().strftime(self.time_format)
                            self.key_string_contamination_20sec = time_start_20sec
                            self.key_string_start_20sec_cycle = "False"
                            self.utility.loginfo(f"CountDown Started For Contamination 20sec")
                        time_now_20sec = (datetime.strptime((str(datetime.now())).split('.')[0], self.time_format))
                        time_diff_20 = (time_now_20sec - datetime.strptime(self.key_string_contamination_20sec,
                                                                           self.time_format)).total_seconds()

                        if time_diff_20 > 0 and self.key_string_freeze_contamination == "True":

                            self.key_string_contamination_material_state = self.key_string_current_material_state
                            self.utility.master_redis.set_val(key=f"{self.key_string_contamination_state_image}",
                                                              val=utility.master_redis.get_val(
                                                                  message.get('inf_op_img_uid')))
                            self.key_string_conta_time_stamp = datetime.now().strftime(self.time_format)
                            self.key_string_freeze_contamination = "False"
                            self.time_tracking_enabled = "True"
                            self.utility.loginfo(
                                f"Freezed Contamination State {self.key_string_contamination_material_state}")


                        elif time_diff_20 >= 5 and self.key_string_prev_event_state == "False":  # and self.time_tracking_enabled == "True":

                            contaminated_img = decode_img(
                                self.utility.master_redis.get_val(f"{self.key_string_contamination_state_image}"))
                            cv2.putText(img=contaminated_img,
                                        text="Contamination Material Started at " + self.utility.master_redis.get_val(
                                            f"{self.key_string_conta_time_stamp}"), org=(200, 175),
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(125, 246, 55),
                                        thickness=2)
                            previous_state_img = decode_img(
                                self.utility.master_redis.get_val(f"{self.key_string_prev_state_image}"))
                            cv2.putText(img=previous_state_img,
                                        text="True Material Started at " + self.utility.master_redis.get_val(
                                            f"{self.key_string_prev_time_stamp}"),
                                        org=(200, 175),
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
                                        color=(125, 246, 55), thickness=2)
                            current_img = decode_img(utility.master_redis.get_val(message.get('inf_op_img_uid')))
                            cv2.putText(img=current_img,
                                        text="Contamination Material Remained" + datetime.strftime(datetime.now(),
                                                                                                   self.time_format),
                                        org=(200, 175),
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=(125, 246, 55),
                                        thickness=2)
                            stacked_img = np.vstack((previous_state_img, contaminated_img, current_img))
                            message['inf_op_img'] = encode_img(stacked_img)
                            message['event'] = "Raw_Material_Contamination"
                            message['current_usecase'] = message['event']
                            message['timestamp'] = datetime.strftime(datetime.now(), self.time_format)
                            message['detected'] = 'Yes'
                            message['severity'] = 'Significant Impact'
                            self.key_string_prev_event_state = "True"
                            self.key_string_freeze_contamination = "True"
                            self.key_string_start_20sec_cycle = "True"
                            self.key_string_start_10min_cycle = "True"
                            self.key_string_prev_material_state = self.key_string_current_material_state
                            self.key_string_prev_state_image = np.zeros([390, 2048])
                            self.key_string_prev_time_stamp = datetime.now().strftime(self.time_format)
                            return message
                        else:
                            return None
                    else:
                        self.key_string_freeze_contamination = "True"
                        self.key_string_start_20sec_cycle = "True"
                else:
                    self.time_tracking_10min_enabled = "False"
                    self.key_string_prev_event_state = "False"
                    self.key_string_freeze_contamination = "True"
                    self.key_string_start_20sec_cycle = "True"
                    self.key_string_start_10min_cycle = "True"
                    self.key_string_prev_material_state = self.key_string_current_material_state
                    self.key_string_prev_time_stamp = datetime.now().strftime(self.time_format)
                    self.utility.master_redis.set_val(key=f"{self.key_string_prev_state_image}",
                                                      val=message.get('inf_op_img'))

            else:
                self.key_string_prev_event_state = "False"
                self.key_string_freeze_contamination = "True"
                self.key_string_start_20sec_cycle = "True"
                self.key_string_start_10min_cycle = "True"
                return None

            except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Exception occurred in MaterialDetection - validate_detection : " + str(err) + ' ' + str(exc_tb.tb_lineno), 'error')






