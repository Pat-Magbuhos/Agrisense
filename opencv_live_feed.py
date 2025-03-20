import cv2

# Open the camera (use 0 for default camera)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 for Raspberry Pi

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    # Show the live camera feed
    cv2.imshow("Live Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
