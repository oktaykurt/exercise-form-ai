import time
import cv2

# Open the video stream
cap = cv2.VideoCapture(0)

time.sleep(4)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

# Get video width, height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# fps = int(cap.get(5))


# # Define the codec and create a VideoWriter object.
# out = cv2.VideoWriter('pushup01.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

# Frame counter
frame_counter = 0

while True:

    ret, frame = cap.read()

    if ret:
        # If the number of frames is less than or equal to 150
        if frame_counter < 15 * 5:
            # Write the frame to the output file
            out.write(frame)

            # Display the resulting frame    
            cv2.imshow('frame', frame)
            print("Frame number", frame_counter)
        else:
            break

        frame_counter += 1
    else:
        break

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
