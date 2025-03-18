import argparse
import pandas
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("parquet_path")
args = parser.parse_args()

data = pandas.read_parquet(args.parquet_path, engine="pyarrow", columns=["robot.linear_velocity"])


vel = np.stack(data['robot.linear_velocity'].to_numpy())


plt.plot(vel[:, 0], 'r-')
plt.plot(vel[:, 1], 'r-')
plt.show()
