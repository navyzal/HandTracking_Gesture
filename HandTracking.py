import socket
import pyautogui
import math
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

leftFingerSign = None
rightFingerSign = None

leftHandIdx = -1
rightHandIdx = -1

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

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    judgeImg = np.zeros((512,512,3), np.uint8)
    leftHandIdx = -1
    rightHandIdx = -1
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
          
          ## 사실은... 왼손임.. 
          if handedness == 'Right':
            
            leftFingerSign = getFingerSign(hand_landmarks.landmark, False)            
            cv2.putText(judgeImg, "Left finger gesture: "  + str(leftFingerSign) , (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            
            '''
            print('left handedness')            
            print('hand_landmark[mp_hands.HandLandmark.THUMB_MCP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x)
            print('hand_landmark[mp_hands.HandLandmark.THUMB_IP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
            print('hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)
            print('hand_landmark[mp_hands.HandLandmark.PINKY_MCP].x', hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)
            '''     
            
          ## 사실은... 오른손임.. 
          elif handedness == 'Left':            
            rightFingerSign = getFingerSign(hand_landmarks.landmark)            
            cv2.putText(judgeImg, "Right finger gesture: "  + str(rightFingerSign) , (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            
            '''
            print('right handedness')            
            print('hand_landmark[mp_hands.HandLandmark.THUMB_MCP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x)
            print('hand_landmark[mp_hands.HandLandmark.THUMB_IP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
            print('hand_landmark[mp_hands.HandLandmark.THUMB_TIP].x', hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)
            print('hand_landmark[mp_hands.HandLandmark.PINKY_MCP].x', hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)
            '''
            
          else:
            cv2.putText(image, "finger gesture: lost", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

      '''
      else:
        cv2.putText(image, "finger gesture: lost", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        '''
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Judgement', judgeImg)
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    #cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
    
    if leftFingerSign == 'Focus' and rightFingerSign == 'Focus':
      print("FOCUS MODE ON")
      # 왼손, 오른손 (실제)
      # 크기 변화 - 양손의 wrist 부터 midFinger 까지의 거리를 구함. 왼손 : 오른손 비율에 따라 이동할 마우스 거리 계산.
      try:
        LeftHandSize = math.dist([results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x,
                  results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y],
                  [results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                  results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        RightHandSize = math.dist([results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST].x,
                  results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST].y],
                  [results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                  results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
      except:
          LeftHandSize = 1
          RightHandSize = 1
      handRatio = 0
      if LeftHandSize > RightHandSize:
        handRatio = -(LeftHandSize / RightHandSize)
      else:
        handRatio = RightHandSize / LeftHandSize
      
      if -1.1 < handRatio < 1.1:
        handRatio = 0
      print(LeftHandSize, RightHandSize, handRatio)
      mouseMoveLength = 10
      
      bytesToSend = str.encode(str(mouseMoveLength * handRatio))
      serverAddressPort = ("192.168.0.171", 20001) 
      bufferSize = 1024 


      # 클라이언트 쪽에서 UDP 소켓 생성 
      UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) 

      # 생성된 UDP 소켓을 사용하여 서버로 전송 

      UDPClientSocket.sendto(bytesToSend, serverAddressPort) 
            
      # Todo: 가속 개념 추가 고려
      pyautogui.moveRel(mouseMoveLength * handRatio, 0)
      
cap.release()
cv2.destroyAllWindows()