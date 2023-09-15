import sys
import numpy as np
import argparse
import yaml
from queue import Queue
import rospy
import cv2
from sensor_msgs.msg import Image

from utils.ros_utils import imgmsg_to_numpy, numpy_to_image
from object_tracker import ObjectTracker
from utils.segment_utils import get_mask_img

img_in_topic = '/acl_jackal/forward/infra1/image_rect_raw'
img_out_topic = '/acl_jackal/segment'

class PromptSelector:
    def __init__(self) -> None:
        self.selecting = False
        self.output_ready = False
        self.windowname = 'select'
        self.frame_time = 30
        self.points = []
        # self.prompts_record = []
        self.present_img = None
        self.present_header = None
        self.present_id = None

        self.frame_que = Queue()
        self.prompt_count = 0

        self.started = False
    def start(self):
        pass

    def draw(self, image, header, img_id):
        if self.selecting:
            print('draw called when selecting !')
            return False
        else:
            self.present_img = image.copy()
            self.present_id = img_id
            cv2.namedWindow(self.windowname, cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(self.windowname, self.mouse_callback)
            cv2.imshow(self.windowname, image)
            key = cv2.waitKey(self.frame_time) & 0xFF
            if key == ord('a') and not self.output_ready:
                print('output ready')
                self.output_ready = True
                self.prompt_count += 1
            elif key != 255:
                print('manual clearing')
                self.clear()
            return True

    def mouse_callback(self, event, x, y, flags, param):
        if self.present_img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.output_ready == False:
            if self.selecting == False:
                self.selecting = True
            cv2.circle(self.present_img, (x,y), 5, (0,0,255), -1)
            cv2.imshow(self.windowname, self.present_img)
            cv2.waitKey(self.frame_time)
            self.points.append((x,y))
            print('img count now:', msg_count)
        # elif event == cv2.EVENT_RBUTTONDOWN:


    def get_output(self):
        if not self.output_ready:
            return None
        
        if len(self.points) == 0:
            out = None
        else:
            points = np.array(self.points)
            labels = np.ones(len(points))
            out = (self.present_id, points, labels)

        self.clear()
        return out

    def clear(self):
        self.points = []
        self.present_img = None
        self.present_id = -1
        self.output_ready = False
        self.selecting = False

data_que = Queue()
selector = PromptSelector()
msg_count = 0
def img_callback(img : Image):
    global msg_count
    image = imgmsg_to_numpy(img)
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))
    selector.frame_que.put((image, img.header, msg_count))
    msg_count += 1

def selector_callback(event):
    while selector.frame_que.empty() == False:
        image, header, msg_count = selector.frame_que.get()
        selector.draw(image, header, msg_count)
        if selector.output_ready:
            output = selector.get_output()
            if output is not None:
                img_id, points, labels = output
                masks = objecet_mapper.generate_mask(image, points, labels)
                if masks.max() == 0:
                    print('mask empty')
                    masks, classes = None, None
                else:
                    classes = np.array([selector.prompt_count])
            else:
                masks, classes = None, None

            print('mask id:', img_id, 'img count:', msg_count)
            data_que.put((image, masks, classes, header))
        else:
            data_que.put((image, None, None, header))


def mapper_callback(event):
    while data_que.empty() == False:
        image, masks_in, classes_in, header = data_que.get()
        out_mask = np.full_like(image, 100)

        if masks_in is not None or msg_count % 2 == 0:
            mapper_out = objecet_mapper.inference_crop(image, masks_in, classes_in)
            if mapper_out is not None:
                _, seg_data_list = mapper_out
                if seg_data_list is not None:
                    masks = seg_data_list[0]['masks']
                    labels = seg_data_list[0]['labels']
                    get_mask_img(masks, labels, out=out_mask)

            mask_msg = numpy_to_image(out_mask, "bgr8")
            mask_msg.header = header
            img_pub.publish(mask_msg)
            continue
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive segmentation node")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = "config/default.yaml"
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int,
        default = 1
    )
    args = parser.parse_args()

    config_file = args.config_file
    print("Load config:", config_file)
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['use_gpu'] = args.gpu

    objecet_mapper = ObjectTracker(configs)
    # img_callback.mapper = objecet_mapper
    selector.start()

    rospy.init_node('object_mapper', anonymous=False)
    mapper_timer = rospy.Timer(rospy.Duration(0.1), mapper_callback)
    selector_timer = rospy.Timer(rospy.Duration(0.01), selector_callback)
    img_sub = rospy.Subscriber(img_in_topic, Image, img_callback)
    img_pub = rospy.Publisher(img_out_topic, Image, queue_size=10)

    rospy.spin()
