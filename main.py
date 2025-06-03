from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture("cars.mp4")
model = YOLO('yolov8n.pt')
names = model.names

#Drawing line
line_y = 850
offset = 15

#Sayac
total_count = 0
counted_ids = set() 


while True: 
    _, frame = cap.read()
    results = model.track(frame, persist = True)
    
    #Line for counting
    start_point = (700, line_y)
    end_point = (1900, line_y)
    cv2.line(frame, start_point, end_point , (0,0,255), 3)

    if results:
        for r in results:
            for box in r.boxes:
            
                #Confidence 
                conf = round(float(box.conf[0]), 2)
            
                #Class Names
                cls = int(box.cls[0])
                class_name = names[cls]

                #Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                

                #Center of cars
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5 ,(255,0,255), -1) 

                #Counting cars
                if class_name in ['car','truck','bus','motorcycle']:
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if 700 < cx < 1900:
                        if (line_y - offset) < cy < (line_y + offset):
                            if track_id != -1 and track_id not in counted_ids:
                                total_count +=1 
                                counted_ids.add(track_id)

                #Label
                cvzone.putTextRect(frame, f'{class_name} {conf}', (x1, y1 - 10), scale=3, thickness=1)
        cvzone.putTextRect(frame, f'Count: {total_count}' , (50,50), scale=3, thickness=3)


    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()