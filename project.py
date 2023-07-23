import cv2

def motion_detection():
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    ret, prev_frame = cap.read()

    while True:
        ret, curr_frame = cap.read()

        if not ret:
            break
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresholded = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(curr_frame)
        cv2.imshow("Motion Detection", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_frame = curr_frame

    cap.release()
    out.release()
    cv2.destroyAllWindows()

motion_detection()
