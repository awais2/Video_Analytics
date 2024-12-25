'''
This script is used the command to openup the multiple camera feed command from a single system:
Command Inputs: Type commands like:
"Show camera 1"
"Show camera 2"
"Exit" to terminate.
'''
import cv2

def show_camera(camera_index):
    try:
        cap = cv2.VideoCapture(camera_index)  # Access the camera by its index
        if not cap.isOpened():
            print(f"Camera {camera_index} is not available.")
            return

        print(f"Showing feed from Camera {camera_index}. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break
            cv2.imshow(f"Camera {camera_index} Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")

def process_command(command):
    if "camera 1" in command.lower() or "webcam" in command.lower():
        show_camera(0)  # Typically, the default webcam is at index 0
    elif "camera 2" in command.lower():
        show_camera(1)  # External camera is usually at index 1
    else:
        print("Command not recognized. Please say 'Show camera 1' or 'Show camera 2'.")

if __name__ == "__main__":
    while True:
        print("Enter your command (or type 'exit' to quit):")
        command = input("> ")  # Command from the operator
        if command.lower() == "exit":
            print("Exiting program.")
            break
        process_command(command)
