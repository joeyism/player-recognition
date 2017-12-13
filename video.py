from player_recognition import add_figures
import cv2

def edit_video(cap):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    counter = 0
    while(True):
        ret, frame = cap.read()
        counter += 1
        print("frame {}".format(counter))
        new_frame = add_figures(frame)
        out.write(new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture("firmino-stoke.mp4.webm")
    edit_video(cap)

