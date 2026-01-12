import pandas as pd
import glob

files = sorted(glob.glob('Fig*_avg_nc_comparison.csv'))
data = {}
for f in files:
    fig_name = f.split('_')[0]
    data[fig_name] = pd.read_csv(f)

print("=== 各方法在每种攻击下的表现 ===\n")
for fig, df in sorted(data.items()):
    print(f"\n{fig}:")
    print(df.to_string(index=False))

print("\n\n=== 各方法在12种攻击类型的平均NC ===")
all_methods = ['Proposed', 'Xi25', 'Wu25', 'Xi24', 'Tan24']
summary_data = {method: [] for method in all_methods}

for fig, df in sorted(data.items()):
    for method in all_methods:
        if method in df.columns:
            avg_nc = df[method].mean()
            summary_data[method].append(avg_nc)
        else:
            summary_data[method].append(None)

summary_df = pd.DataFrame(summary_data, index=sorted(data.keys()))
print(summary_df.to_string())

print("\n=== 各方法总平均NC ===")
total_avg = summary_df.mean()
print(total_avg.to_string())
