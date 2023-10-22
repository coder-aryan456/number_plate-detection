import os
import random
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
import time
from util import get_car, read_license_plate, write_csv
from tracker import Tracker
from PIL import Image


video_path = os.path.join('.', 'data', 'carvid_cut.mp4')
video_out_path = os.path.join('.', 'out.mp4') 

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

#code to write output video in folder
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

#requiring yolo model for car detection
model = YOLO("yolov8x.pt")
license_plate_detector = YOLO('license_plate_detector.pt')

#requiring deepsort
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
count=0
data_deque={}
trailslen=10
time_start={}
time_end={}
speed={}
number_of_frames={}
# class_id from coco sheet
vehicles = [2, 3, 5, 7]
frame_nmr=-1
text_license={}
prev_frame_time =0
new_frame_time=0
temp_license_plate_text={}
photo={}
while True:
    ret,frame = cap.read()
    if ret :
# if you want to skip frames to reduce processing        
        count +=count
        if count%6!=0:
            continue


        #detection of vehicles
        frame_nmr +=1
       
        results = model(frame,show=True)[0]
#bounding area for estimation of speed
        line = np.array([(375,660), (1253,660),(1585,724),(95,700)], dtype=np.int32)

        line = line.reshape((-1, 1, 2))

# Draw the polygon on the image for speed calculation
        cv2.polylines(frame, [line], isClosed=True, color=(0, 255, 0), thickness=2)


            
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in vehicles :
                if score >0.5:
                    detections.append([x1, y1, x2, y2, score])

#tracking of vehicles deepSort
        tracker.update(frame, detections)
       
#speed calculation code start form here
        count1=0
        id_present={}
        elements_for_licence=[]
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            elements=np.array([x1,y1,x2,y2,track_id])
            elements_for_licence.append(elements)
            
            track_id_string=str(track_id)

#test if center point of vehicle is inside or outside the polygon created on video
            distance1 = cv2.pointPolygonTest(line, (int((x2+x1)/ 2), int((y1+y2)/2)), True)

            
            if track_id not in speed:
                speed[track_id]=(-1,False)
                number_of_frames[track_id]=(0,True)
            speed_tuple=speed[track_id]
            temp=number_of_frames[track_id]
#if car inters the polygon and inside the polygon then distance should be >0
            if distance1>=0 and not speed_tuple[1]:
                val=time.time()
                # print(val)
                time_start[track_id]=val
                speed[track_id]=(-1,True)
                temp=(0,False)
            
                
#after leaving the polygon
            elif distance1 <0 and speed_tuple[1]:
                time_end[track_id]=time.time()
                # print(time_end[track_id])
                # s=(time_end[track_id]-time_start[track_id])
                t_req=temp[0]/30
                s=(20*(18/5))/t_req
                speed[track_id]=(s,False)
                print(number_of_frames[track_id],"this is number of frames taking by car id", track_id)
                val2=temp[0]
                temp=(val2,True)
                number_of_frames[track_id]=(val2,True)

            if not temp[1]:
                val1=temp[0]
                val1=val1+1
                number_of_frames[track_id]=(val1,False)
            
            speed_of_car=str(speed_tuple[0])
            frame_number=str(number_of_frames[track_id])
            cv2.putText(frame, "frame_number = "+frame_number+" "+speed_of_car,(int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            id_present[track_id]=(int((x2+x1)/ 2), int((y1+y2)/2))
        track_np_array=np.array(elements_for_licence)

 # detect license plates thourgh yolo
        license_plates = license_plate_detector(frame,show=True)[0]
        for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

# assign license plate to car to insure which license plate belong to which car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_np_array)
                # print(car_id,"car_id")
                if car_id != -1:

# crop license plate 
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

# process license plate...convert it into grey scale to reduce processing
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    
                    
 # read license plate number through easyocr
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                   
                    temp_license_plate_text[car_id]=(license_plate_text,license_plate_text_score)
# saving image in images_generated directory only if car_speed is greater than 30
                    if car_id not in photo:
                        photo[car_id]=(speed[car_id][0],False)
                    else: 
                       if speed[car_id][0]>=20 and not photo[car_id][1]:
                            photo[car_id]=(speed[car_id][0],True)
                            name=str(frame_nmr)+str(car_id)+".png"
                            cv2.imwrite('images_generated/' + name,license_plate_crop)
                    # cv2.waitKey(0)
#saving all images regardless of speed in all_images_generated_directory
                    name=str(frame_nmr)+str(car_id)+".png" 
                    cv2.imwrite('all_images_generated/' + name,license_plate_crop)
                   
                    if temp_license_plate_text[car_id][0] is not None:
                        text_license[car_id] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                  'text': temp_license_plate_text[car_id][0],
                                                                  'bbox_score': score,     
                                                                  'text_score': temp_license_plate_text[car_id][1],
                                                                  'speed':speed[car_id][0]}}    
                                                                      



        #check if car is in frame or not
        for key in list(data_deque):
            if key not in id_present:
                data_deque.pop(key)
                speed.pop(key)
                # time_start.pop(key)
                # time_end.pop(key)
                # photo.pop(key)

        for key1 in id_present:
            if key1 not in data_deque:  
                data_deque[key1] = deque(maxlen=trailslen)
                data_deque[key1].appendleft(id_present[key1])
            else:
                data_deque[key1].appendleft(id_present[key1])
                for i in range(1, len(data_deque[key1])):
            # check if on buffer value is none
                    if data_deque[key1][i - 1] is None or data_deque[key1][i] is None:
                        continue
            # generate dynamic thickness of trails
                    thickness = int(np.sqrt(trailslen / float(i + i)) * 5)
                # draw trails
                    # cv2.line(frame, data_deque[key1][i - 1], data_deque[key1][i], (255,0,0),thickness)


        cap_out.write(frame)
        cv2.imshow('Frame',frame)

    else:
     break
# write_csv(text_license, 'test.csv')
# print(text_license)
write_csv(text_license, 'test.csv')

# print("text of the liscense")
# cap.release()
# # cap_out.release()
# cv2.destroyAllWindows()
