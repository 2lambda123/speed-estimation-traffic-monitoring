import configparser
import json

import pandas as pd


def analyzer(log_path):
    config = configparser.ConfigParser()
    config.read("speed_estimation/config.ini")

    # add a new section and some values
    try:
        config.add_section("analyzer")
    except:
        print("")

    log_dict = []
    speed_limit = int(config.get("analyzer", "speed_limit"))
    # speed_limit = 80

    print(log_path)

    with open(log_path, "r", encoding='UTF-8') as log_file:
        for _, line in enumerate(log_file):
            if not line.startswith("INFO:root:{"):
                continue
            line_dict = json.loads(line[10:])
            if "car_id" in line_dict:
                log_dict.append(line_dict)

        data_frame = pd.DataFrame(log_dict)

        df_grouped = data_frame.groupby("car_id").agg(
            direction_indicator=("direction_indicator", "sum"),
            frame_count=("frame_count", "count"),
        )

        # outlier filtering
        df_grouped = df_grouped[df_grouped["frame_count"] > 20]
        df_grouped = df_grouped[df_grouped["direction_indicator"] != 0]  # both directions

        avg_frame_count = df_grouped["frame_count"].mean()

        config.set("analyzer", "avg_frame_count", str(avg_frame_count))

        speeds = []

        for _, row in df_grouped.iterrows():
            speed_for_car = (avg_frame_count / row["frame_count"]) * speed_limit
            speeds.append(speed_for_car)
            # print(str(index) + ": " + str(speed_for_car))

        df_grouped["speed"] = speeds
        print(df_grouped)

        with open("speed_estimation/config.ini", "w") as configfile:
            config.write(configfile)

        return avg_frame_count, speed_limit