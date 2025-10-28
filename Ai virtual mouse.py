import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

def finger_state(lm_list):
    """Return list of which fingers are up (1=up, 0=down)."""
    fingers = []
    # Thumb
    if lm_list[4][0] < lm_list[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if lm_list[tip][1] < lm_list[pip][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [thumb, index, middle, ring, pinky]


prev_y = None
scroll_mode = False  # whether we are currently scrolling
scroll_direction = None  # 'up' or 'down'
last_scroll_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                fingers = finger_state(lm_list)
                total_fingers = fingers.count(1)
                index_tip = lm_list[8]
                mouse_x = int((index_tip[0] / w) * screen_w)
                mouse_y = int((index_tip[1] / h) * screen_h)

                # 🖐️ Move Mouse
                if total_fingers == 5:
                    pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)
                    cv2.putText(frame, "Move Mouse", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                    scroll_mode = False

                # ☝️ One finger = Left Click
                elif fingers[1] == 1 and total_fingers == 1:
                    pyautogui.click()
                    cv2.putText(frame, "Left Click", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    scroll_mode = False

                # ✋ 3 fingers = Drag
                elif total_fingers == 3:
                    pyautogui.mouseDown()
                    cv2.putText(frame, "Drag", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    scroll_mode = False

                # 🤚 3 fingers closed = Drop
                elif total_fingers == 0:
                    pyautogui.mouseUp()
                    cv2.putText(frame, "Drop", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    scroll_mode = False

                # ✌️ Two fingers (index + middle)
                elif fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2:
                    index_tip = lm_list[8]
                    middle_tip = lm_list[12]
                    distance = ((index_tip[0] - middle_tip[0]) ** 2 +
                                (index_tip[1] - middle_tip[1]) ** 2) ** 0.5

                    # Two fingers separate → right click
                    if distance > 60:
                        pyautogui.rightClick()
                        cv2.putText(frame, "Right Click", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        scroll_mode = False

                    # Two fingers close together → continuous scroll
                    else:
                        avg_y = (index_tip[1] + middle_tip[1]) // 2
                        if prev_y is not None:
                            if avg_y < prev_y - 20:
                                scroll_direction = 'up'
                            elif avg_y > prev_y + 20:
                                scroll_direction = 'down'
                        prev_y = avg_y
                        scroll_mode = True
                        cv2.putText(frame, "Scrolling Mode", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    else:
        scroll_mode = False  # stop scroll if hand not detected

    # Continuous scroll
    if scroll_mode and scroll_direction:
        now = time.time()
        if now - last_scroll_time > 0.1:  # adjust speed
            if scroll_direction == 'up':
                pyautogui.scroll(200)
            elif scroll_direction == 'down':
                pyautogui.scroll(-200)
            last_scroll_time = now

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
