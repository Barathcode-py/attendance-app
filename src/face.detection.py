import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor_path =  "C:\\Users\\USER\\attendance-app\\src\\shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(predictor_path)



# Load the face detector
detector = dlib.get_frontal_face_detector()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Draw rectangles around faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
