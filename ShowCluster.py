import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# # Before clustering
# df = pd.read_csv("./output_serial.csv", header=None)
# df.columns = ["x", "y","cluster"]
# sns.scatterplot(x=df["x"], 
#                 y=df["y"])
# plt.title("Scatterplot of Clustered")

# After clustering
plt.figure()
df1 = pd.read_csv("./output_point_serial.csv")
df2 = pd.read_csv("./output_centro_serial.csv")
merged_df = pd.merge(df1, df2, left_on='c', right_index=True)
# Add a new column 'label' to df1 with values from df2
df1['label'] = merged_df['y']
sns.scatterplot(x=df1.x, y=df1.label, 
                hue=df1.c, 
                palette=sns.color_palette("hls", n_colors=5))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Clustered")

plt.show()