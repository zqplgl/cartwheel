import _IObjZoneDetect as od
import os
import cv2
import numpy as np
from py_cpu_nms import py_cpu_nms

class IObjZoneYOLOV3Detect:
    def __init__(self, cfg_file, weight_file, gpu_id=0):
        self.__detector = od.ObjZoneDetector(cfg_file,weight_file,0,0)

    def detect(self,im,confidence_thread = 0.7):
        r = []
        boxes = self.__detector.detect(im,im.shape[1],im.shape[0],confidence_thread)
        for b in boxes:
            temp = {}
            temp["zone"] = (b.zone.x,b.zone.y,b.zone.x+b.zone.w,b.zone.y+b.zone.h)
            temp["cls"] = b.cls
            temp["score"] = b.score
            r.append(temp)
        return r


class ICartwheelZoneDetect:
    def __init__(self,model_dir, gpu_id=0):
        if not model_dir.endswith("/"):
            model_dir += "/"
        cfg_file = model_dir + "cartwheel/yolov3.cfg"
        weight_file = model_dir + "cartwheel/yolov3.weights"
        self.__detector = IObjZoneYOLOV3Detect(cfg_file,weight_file,gpu_id)

    def detect(self, im, confidence_threshold=0.7, nms_threshold=0.1):
        boxes = self.__detector.detect(im,confidence_threshold)
        if(len(boxes)<1):
            return 0, []
        tmp_boxes = []
        for box in boxes:
            tmp_boxes.append((box["zone"][0],box["zone"][1],box["zone"][2],box["zone"][3],box["score"], box["cls"]))

        tmp_boxes = np.array(tmp_boxes)
        keep = py_cpu_nms(tmp_boxes, nms_threshold)
        tmp_boxes = []
        for num in keep:
            tmp_boxes.append(boxes[num])
        boxes = tmp_boxes

        result = 0
        for box in boxes:
            result += (box["cls"]+1)

        return result,boxes

def addRectangle(im,boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        zone = box["zone"]
        cv2.rectangle(im,(zone[0],zone[1]),(zone[2],zone[3]),(0,0,255),2)
        cv2.putText(im, "%s:%.2f"%(box["cls"],box["score"]),(zone[0],zone[1]),font,1,(0,255,0))

def run():
    model_dir = "/root/models/"
    detector = ICartwheelZoneDetect(model_dir)

    picdir = "/root/testpic/cartwheel/"
    for picname in os.listdir(picdir):
        im = cv2.imread(picdir+picname)
        result,boxes = detector.detect(im)
        addRectangle(im, boxes)
        print(result)

if __name__=="__main__":
    run()


