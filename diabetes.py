import cv2
import mediapipe as mp
import math as m
## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
name='Sayedmulla_Trim'+'.mp4'
cap = cv2.VideoCapture(name)
distances = []
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

while cap.isOpened():
    # read frame
    _,frame = cap.read()
    try:
        # resize the frame for portrait video
        frame = cv2.resize(frame, (1000,500))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get height and width.
        h, w = frame_rgb.shape[:2]
        
        # process the frame for pose detection
        pose_results = pose.process(frame_rgb)
        # print(pose_results.pose_landmarks)
        
#         # Use lm and lmPose as representative of the following methods.
        lm = pose_results.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        l_heel_x = int(lm.landmark[lmPose.LEFT_HEEL].x * w)
        l_heel_y = int(lm.landmark[lmPose.LEFT_HEEL].y * h)
        
        
        r_heel_x = int(lm.landmark[lmPose.RIGHT_HEEL].x * w)
        r_heel_y = int(lm.landmark[lmPose.RIGHT_HEEL].y * h)
        distances.append(findDistance(l_heel_x, l_heel_y, r_heel_x, r_heel_y))
#         # Acquire the landmark coordinates.
#         # Once aligned properly, left or right should not be a concern.      
#         # Left shoulder.
#         l_shldr_x = int(lm.landmark[lmPose.LEFT_HEEL].x * w)
#         l_shldr_y = int(lm.landmark[lmPose.LEFT_HEEL].y * h)
#         # Right shoulder
#         r_shldr_x = int(lm.landmark[lmPose.RIGHT_HEEL].x * w)
#         r_shldr_y = int(lm.landmark[lmPose.RIGHT_HEEL].y * h)

#         # Calculate distance between left shoulder and right shoulder points.
#         offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
#         cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
#         cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
#         cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
#         cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        
#         # Join landmarks.
#         cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
#         cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        
        # draw skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # display the frame
        cv2.imshow('Output', frame)
#         cv2.waitKey(0)
    except Exception as e:
        pass
#         print(e)
#         break
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# avg = sum(distances)/len(distances)
avg = max(distances)
print(f"Distance between feet = {avg}")
print("Diabetic" if avg < 150 else "Normal")