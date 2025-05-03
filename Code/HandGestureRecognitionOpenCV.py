import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Failed to capture frame")
        continue

    # Draw ROI rectangle
    cv2.rectangle(img, (50, 50), (400, 400), (0, 255, 0), 2)
    crop_img = img[50:400, 50:400]

    # Preprocessing
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (35, 35), 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Thresholded', thresh1)

    # Contours
    contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        cv2.imshow('Gesture', img)
        if cv2.waitKey(1) == 27:
            break
        continue

    cnt = max(contours, key=cv2.contourArea)

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convex hull
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)

    count_defects = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # Lengths of triangle sides
            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)

            if b * c == 0:
                continue

            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / math.pi)

            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 5, [0, 0, 255], -1)

            cv2.line(crop_img, start, end, [0, 255, 0], 2)

    # Gesture output
    if count_defects == 1:
        gesture_text = "GESTURE ONE"
    elif count_defects == 2:
        gesture_text = "GESTURE TWO"
    elif count_defects == 3:
        gesture_text = "GESTURE THREE"
    elif count_defects == 4:
        gesture_text = "GESTURE FOUR"
    else:
        gesture_text = "Hello World!!!"

    cv2.putText(img, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # Display results
    cv2.imshow('Gesture', img)
    combined = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', combined)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
