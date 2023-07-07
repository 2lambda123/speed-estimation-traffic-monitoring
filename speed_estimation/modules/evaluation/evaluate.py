import json
import os
import re
import uuid

import numpy as np
import pandas as pd
import plotly.express as px


def load_log(log_path: str):
    cars_path = None
    max_depth = 0
    avg_speeds = []

    with open(log_path, "r") as fp:
        for idx, line in enumerate(fp):
            if idx == 0:
                result = re.search("Video: (.*)", line)
                cars_path = result.group(1)
                print(f"Found cars path from log: {cars_path}")
            if not line.startswith("INFO:root:{"):
                continue
            line_dict = json.loads(line[10:])
            if "avgSpeedTowards" in line_dict:
                avg_speeds.append(line_dict)

    video_id = cars_path.strip("/").split("/")[-1]
    estimation = pd.DataFrame(avg_speeds)
    # estimation.rename({'avgSpeedLastMinute': f"{video_id}_depth_{max_depth}"}, inplace=True)
    return (f"{video_id}_depth_{max_depth}", estimation, cars_path)


def avg_speed_for_time_ground_truth(cars_truth, timeStart, timeEnd):
    cars_to_avg = cars_truth.loc[
        cars_truth["start"].gt(timeStart) & cars_truth["end"].le(timeEnd)
    ]
    return cars_to_avg["speed"].mean()


def avg_speed_for_time_estimation(estimation, timeStart, timeEnd):
    timeStart *= 50
    timeEnd *= 50
    estimation_avg = estimation.loc[
        estimation["frameId"].gt(timeStart) & estimation["frameId"].le(timeEnd)
    ]
    return estimation_avg["avgSpeedTowards"].mean()


def generate_aligned_estimations(run_ids, loaded_avg_speeds, ground_truth):
    truth = []
    estimations = {k: [] for k in run_ids}
    timestamps = []

    for start in range(300, 30 * 60, 60):
        end = start + 60
        truth.append(avg_speed_for_time_ground_truth(ground_truth, start, end))
        for idx, id in enumerate(run_ids):
            estimations[id].append(
                avg_speed_for_time_estimation(loaded_avg_speeds[idx], start, end)
            )
        timestamps.append(end)

    return truth, estimations, timestamps


def plot_absolute_error(logs: "list[str]", save_file_path=None):
    run_ids, loaded_avg_speeds, cars_paths = zip(*list(map(load_log, logs)))
    # Check that all logs are from the same video
    cars_path = cars_paths[0]
    for c in cars_paths:
        if c != cars_path:
            raise Exception("Can only evaluate logs of the same video in one call!")

    cars = pd.read_csv(cars_path + "cars.csv")
    truth, estimations, timestamps = generate_aligned_estimations(
        run_ids, loaded_avg_speeds, cars
    )

    video_id = cars_path.strip("/").split("/")[-1]
    # Calculate values per minute
    df = pd.DataFrame({"truth": truth, **estimations, "timestamps": timestamps})
    # Remove timestamps where no estimations or truth is available
    df.dropna(axis=0, inplace=True)
    fig = px.line(
        df, x="timestamps", y=df.columns, title=f"Absolute Estimations ({cars_path})"
    )
    id = uuid.uuid4().hex[:10]
    if save_file_path is not None:
        fig.write_image(
            file=os.path.join(save_file_path, f"{video_id}_{id}_estimations.pdf")
        )
    else:
        fig.show()

    run_id_list = list(run_ids)
    df[run_id_list] = df[run_id_list].sub(df["truth"], axis=0)
    fig = px.line(
        df, x="timestamps", y=df.columns[1:], title=f"Mean Absolute Error ({cars_path})"
    )
    if save_file_path is not None:
        fig.write_image(file=os.path.join(save_file_path, f"{video_id}_{id}_mae.pdf"))
    else:
        fig.show()

    if save_file_path is not None:
        csv_path = os.path.join(save_file_path, f"{video_id}_{id}_error.csv")
        df[run_id_list].mean(axis=0).to_csv(csv_path)


def get_gt_for_cctv_dataset(path):
    # 0: towards camera, 1: away from camera
    lanes_to_use = [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
    ]
    fps = 25

    summary = []
    count = 0

    for file in sorted(os.listdir(path)):
        filename = os.fsdecode(file)
        file_number = filename.split(".")[0]
        n = np.load(os.path.join(path, filename))

        speed = round(np.mean(n["speeds"]), 2)
        frames_seen = n["timestamps"][len(n["timestamps"]) - 1] - n["timestamps"][0]

        gt = {
            "filename": file_number,
            "speed_in_kmh": speed,
            "time_seen": frames_seen / fps,
            "lane_to_use": lanes_to_use[count],
        }

        summary.append(gt)
        count += 1

    return summary


def main():
    arr = ["/home/ssawicki/porsche_digital_hpi/logs/20230204-090410_run_371dbc8dd5.log"]

    plot_absolute_error(arr, "logs/")

    gf_cctv = get_gt_for_cctv_dataset(
        "/Users/matthiasschneider/Downloads/cctv_dataset/data/cctv_videos/gt"
    )


if __name__ == "__main__":
    main()
