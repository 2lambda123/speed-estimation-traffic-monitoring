import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def run():
    cap = cv2.VideoCapture('../datasets/switch_dataset.mp4')
    # Initialize count
    count = 0

    frames = []
    percentage_of_zeros = 0

    full_changes = []
    cont = "not moving"
    cont_hard = "no hard move happened"
    q1 = 0.4
    while True:
        ret, frame = cap.read()
        count += 1
        cv2.putText(frame, str(percentage_of_zeros), (30, 30), 0, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(cont), (30, 80), 0, 1, (0, 0, 255), 2)

        cv2.putText(frame, str(cont_hard), (30, 150), 0, 1, (127, 255, 0), 2)


        cv2.putText(frame, str(q1), (30, 120), 0, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        # print(frame)
        # print("NEXT------------")
        frames.append(frame)
        if len(frames) == 2:
            out = frames[0] - frames[1]
            # print(out)

            out[-11:11] = 0 # try to remove random noise

            zeros = out.size - np.count_nonzero(out)
            size = out.size

            percentage_of_zeros = zeros/size
            # print(percentage_of_zeros)

            full_changes.append(percentage_of_zeros)

            frames = []

            starter_threshold = 0.4
            q1 = np.percentile(full_changes, 25)
            if len(full_changes) >= 100:
                starter_threshold = q1




            if percentage_of_zeros < starter_threshold:
                cont = ("probably slightly moving right now")
                #print(cont)
            else:
                cont = ("probably NOT moving right now")
                #print(cont)

            if percentage_of_zeros < starter_threshold/4: #divided by 4 for hard move
                cont = ("probably HARD MOVE HAPPENED")
                cont_hard = cont +" at frame: " + str(count)

                full_changes = []
                print(cont_hard)

            # break

        if len(full_changes) == 400:
            avg_changes = sum(full_changes)/len(full_changes)
            #print("avg")
            #print(avg_changes)
            q1 = np.percentile(full_changes, 25)

            # print("1Q")
            # print(q1)
            # plt.hist(full_changes)
            # plt.show()
            lenght_f = len(full_changes)
            last100 = lenght_f - 102
            del full_changes[last100:]


        if not ret:
            break








        key = cv2.waitKey(1)
        if key == 27:

            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()

