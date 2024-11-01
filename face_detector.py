import cv2
import cv2.data

# Path to the default Haar Cascade for face detection
path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Use OpenCV's built-in classifier
cameraNo = 0  # Camera number (try 0 if 1 doesn't work)
objectName = 'FACE'  # Object name to display
frameWidth = 640  # Display width
frameHeight = 480  # Display height
color = (255, 0, 255)  # Color for the rectangle (BGR format)


# Load Haar Cascade classifier
cascade = cv2.CascadeClassifier(path)

# Initialize camera
cap = cv2.VideoCapture(cameraNo)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Empty callback function for trackbars
def empty(a):
    pass

# Create Trackbars
cv2.namedWindow("Result")
cv2.resizeWindow("Result", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neig", "Result", 8, 20, empty)
cv2.createTrackbar("Min Area", "Result", 0, 100000, empty)
cv2.createTrackbar("Brightness", "Result", 100, 255, empty)

while True:
    # Set camera brightness from trackbar value
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)

    # Get camera image and convert to grayscale
    success, img = cap.read()
    if not success:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the object using the cascade
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Result") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray, scaleVal, neig)

    # Display the detected objects
    for (x, y, w, h) in objects:
        area = w * h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

    # Show the result
    cv2.imshow("Result", img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
