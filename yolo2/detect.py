'''
Created on Aug 5, 2018
Reference from marvis, but this version does not use cuda
look through the website if you want to know more
https://github.com/marvis/pytorch-yolo2
'''

import torch
from PIL import Image, ImageDraw
import cv2

from darknet import DarkNet
from utils import *


def detect(cfgfile, weightfile, imgfile):
    """
    - weight file was downloaded  at https://pjreddie.com/darknet/yolov2/
    :cfgfile    network config file
    :weightfile weight file, include w and b
    :imgfile    single image input
    """
    # init model
    model = DarkNet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    print 'load weights done!'
    if model.num_class == 20:
        namesfile = 'cfg/voc.names'
    elif model.num_class == 80:
        namesfile = 'cfg/coco.names'
    img = Image.open(imgfile).convert('RGB')
    # resize to 416x416
    img_sized = img.resize((model.width, model.height))
    # get model output
    output = get_net_output(model, img_sized)
    # get region boxes
    boxes = get_region_boxes(output, 0.5, model.num_class, model.anchors, model.num_anchor)[0]
    # iou > 0.4
    boxes = nms(boxes, 0.4)
    for i in range(len(boxes)):
        print boxes[i]
    class_names = load_class_names(namesfile)
    # plot bounding box
    plot_boxes(img, boxes, 'predictions.jpg', class_names)
    # show img
    predictions_img = Image.open('predictions.jpg')
    predictions_img.show()


def get_net_output(model, img):
    model.eval() # test model
    # img to tensor
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0, 2).contiguous()
        img = img.view(1, 3, width, height)
        img = img.float().div(255.0)
    # cv2 image use camera
    elif type(img) == np.ndarray: 
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)
    img = torch.autograd.Variable(img)
    # get model output
    output = model(img)
    # output(1, 425, 13, 13)
    output = output.data
    return output

def camera_detect(cfgfile, weightfile):
    """
    - camera detect
    :cfgfile    use tiny config file
    :weightfile use tiny weight file 
    """
    model = DarkNet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    print 'load weights done!'
    if model.num_class == 20:
        namesfile = 'cfg/voc.names'
    elif model.num_class == 80:
        namesfile = 'cfg/coco.names'
    class_names = load_class_names(namesfile) 
    cap = cv2.VideoCapture(0)   
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    while True:
        res, img = cap.read()
        if res:
            img_sized = cv2.resize(img, (model.width, model.height))  
            # get output
            output = get_net_output(model, img_sized)
            boxes = get_region_boxes(output, 0.5, model.num_class, model.anchors, model.num_anchor)[0]
            print 'boxes:', boxes
            boxes = nms(boxes, 0.4)
            print('-------------')
            draw_img = plot_boxes_cv2(img, boxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            cv2.waitKey(1)
        else:
             print("Unable to read image")
             exit(-1) 
    
def run_img():
    cfgfile = 'cfg/yolo.cfg'
    weightfile = 'weight/yolo.weights'
    imgfile = 'data/person.jpg'
    detect(cfgfile, weightfile, imgfile)

def run_cam():
    cfgfile = 'cfg/yolov2-tiny-voc.cfg'
    weightfile = 'weight/yolov2-tiny-voc.weights'
    camera_detect(cfgfile, weightfile)

if __name__ == '__main__':
    #run_cam()
    cfgfile = 'cfg/yolo.cfg'
    weightfile = 'weight/yolo.weights'
    camera_detect(cfgfile, weightfile)
    
    
    