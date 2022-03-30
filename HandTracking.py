import socket
import time
#import pyautogui as pi
import pydirectinput as pi
pi.PAUSE = 0.01
import math
import cv2
import mediapipe as mp
import numpy as np
import vgamepad as vg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

THRESHOLD_CHECK_HANDS_FPS = 5
THRESHOLD_HAND_SIZE_SENSIBILITY = 1      # 2이면 두 손의 크기 차이가 1.2 배 이상일때 감지

# Virtual GamePad
gamepad = vg.VX360Gamepad()

# For Camera input:
cap = cv2.VideoCapture(1)    # USB WebCam
#cap = cv2.VideoCapture('http://test:0317047370@192.168.0.127:8080/videofeed')   # IP Cam

def inputDirectionKey(handRatio):
  if handRatio > 0:
    pi.press('right')
  elif handRatio < 0:
    pi.press('left')

def getRotation(handProcessResults, sensibility=2):
  try:
    LeftHandSize = math.dist([handProcessResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x,
              handProcessResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y],
              [handProcessResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
              handProcessResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
    RightHandSize = math.dist([handProcessResults.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST].x,
              handProcessResults.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST].y],
              [handProcessResults.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
              handProcessResults.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
  except:
      LeftHandSize = 1
      RightHandSize = 1
  
  if LeftHandSize <= 0:
    LeftHandSize = 1
  if RightHandSize <= 0:
    RightHandSize = 1
  
  handRatio = 10
  rotation = 0

  # 양손의 비율을 구한 후 1.58 이면 최종 값을 5, -1.43 이면 -4 로 나오게 처리...
  try:
    if LeftHandSize > RightHandSize: 
      rotation = -1 * math.trunc(LeftHandSize / RightHandSize * 10) + 10
    elif LeftHandSize < RightHandSize:
      rotation = math.trunc(RightHandSize / LeftHandSize * 10) - 10
      
  except:
    rotation = 0
        
  print(rotation)
  if rotation > 20:
    rotation = 20
  elif rotation < -20:
    rotation = -20
  elif -(sensibility) < rotation < sensibility:      # 손 흔들림에 따른 민감도 제어값
    rotation = 0
  
  if rotation < 0:
    rotation = rotation*500 - 8000
  elif rotation > 0:
    rotation = rotation*500 + 8000

  return rotation

def getFingerSign(hand_landmark, isRightHand=True):        
    ret = None
    thumbIsOpen = False
    indexFingerIsOpen = False
    middleFingerIsOpen = False
    ringFingerIsOpen = False
    pinkyIsOpen = False
    isShowingPalm = False
   
    pseudoFixKeyPoint = hand_landmark[mp_hands.HandLandmark.THUMB_MCP].x
    
    if isRightHand:
      if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x > hand_landmark[mp_hands.HandLandmark.PINKY_MCP].x:
        isShowingPalm = True
      
        if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x > pseudoFixKeyPoint \
          and hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x > pseudoFixKeyPoint:
          thumbIsOpen = True
      else:
        if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x < pseudoFixKeyPoint \
          and hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x < pseudoFixKeyPoint:
          thumbIsOpen = True
          
    else:          
      if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x < hand_landmark[mp_hands.HandLandmark.PINKY_MCP].x:
        isShowingPalm = True      
        
        if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x < pseudoFixKeyPoint \
          and hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x < pseudoFixKeyPoint:
            thumbIsOpen = True
      else:
        if hand_landmark[mp_hands.HandLandmark.THUMB_IP].x > pseudoFixKeyPoint \
          and hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x > pseudoFixKeyPoint:
            thumbIsOpen = True  


    pseudoFixKeyPoint = hand_landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    if hand_landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y < pseudoFixKeyPoint and hand_landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < pseudoFixKeyPoint:
        indexFingerIsOpen = True

    pseudoFixKeyPoint = hand_landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    if hand_landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y < pseudoFixKeyPoint and hand_landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < pseudoFixKeyPoint:
        middleFingerIsOpen = True

    pseudoFixKeyPoint = hand_landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    if hand_landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y < pseudoFixKeyPoint and hand_landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < pseudoFixKeyPoint:
        ringFingerIsOpen = True

    pseudoFixKeyPoint = hand_landmark[mp_hands.HandLandmark.PINKY_PIP].y
    if hand_landmark[mp_hands.HandLandmark.PINKY_DIP].y < pseudoFixKeyPoint and hand_landmark[mp_hands.HandLandmark.PINKY_TIP].y < pseudoFixKeyPoint:
        pinkyIsOpen = True
    
    if (hand_landmark[mp_hands.HandLandmark.WRIST].y - hand_landmark[mp_hands.HandLandmark.PINKY_TIP].y) > 0 and not thumbIsOpen and not indexFingerIsOpen and not middleFingerIsOpen and not ringFingerIsOpen and not pinkyIsOpen:        
        ret = 'ROCK'
    elif thumbIsOpen and indexFingerIsOpen and middleFingerIsOpen and ringFingerIsOpen and pinkyIsOpen and isShowingPalm:        
        ret = 'Focus'
    elif thumbIsOpen and indexFingerIsOpen and middleFingerIsOpen and ringFingerIsOpen and pinkyIsOpen:        
        ret = 'FIVE'
    elif not thumbIsOpen and indexFingerIsOpen and not middleFingerIsOpen and not ringFingerIsOpen and not pinkyIsOpen:        
        ret = 'ONE'
    else:        
        '''
        print("thumbIsOpen, indexFingerIsOpen, middleFingerIsOpen, ringFingerIsOpen, pinkyIsOpen")
        print(thumbIsOpen, indexFingerIsOpen, middleFingerIsOpen, ringFingerIsOpen, pinkyIsOpen)        
        '''      
        pass

    # if ret is not None: print('finger gesture = ', ret)
    return ret

prev_time = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # FPS Check
    start = time.time()
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = image.copy()    
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
    results = hands.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check Hands every {THRESHOLD_CHECK_HANDS_FPS}
    current_time = time.time() - prev_time
    if current_time < (1 / THRESHOLD_CHECK_HANDS_FPS):
      continue
    
    prev_time = time.time()
    
    leftFingerSign = None
    rightFingerSign = None
    
    # Get FingerSign for both hands
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
          handedness = results.multi_handedness[idx].classification[0].label
                    
          if handedness == 'Right':
            leftFingerSign = getFingerSign(hand_landmarks.landmark, False)            
            cv2.putText(image, "Right finger gesture: "  + str(leftFingerSign) , (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          
          elif handedness == 'Left':            
            rightFingerSign = getFingerSign(hand_landmarks.landmark)            
            cv2.putText(image, "Left finger gesture: "  + str(rightFingerSign) , (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
          else:
            cv2.putText(image, "finger gesture: lost", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # FPS Check
    end = time.time()
    totalTime = end - start
        
    if totalTime != 0:
      fps = 1 / totalTime  
      cv2.putText(image, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    if leftFingerSign == 'Focus' and rightFingerSign == 'Focus':
      print("FOCUS MODE ON")

      rotation = getRotation(results)

        
      print(f'rotation: {rotation}')
      cv2.putText(image, f'rotation: {rotation}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      
      #inputDirectionKey(handRatio)
      print(rotation)
      if rotation != 0:
        gamepad.right_joystick(x_value=rotation, y_value=0)  # values between -32768 and 32767
      else:
        gamepad.right_joystick(x_value=0, y_value=0)  # values between -32768 and 32767
      gamepad.update()
    
    else:
      gamepad.right_joystick(x_value=0, y_value=0)  # values between -32768 and 32767
      gamepad.update()
        
    cv2.imshow('MediaPipe Hands', image)
        
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap.release()
cv2.destroyAllWindows()