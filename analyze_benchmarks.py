import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def read_params_and_summary(base_dir):
    data = []

    for root, dirs, files in os.walk(base_dir):
        # If the relative dir starts with __, skip it
        if os.path.relpath(root, base_dir).startswith("__") or os.path.relpath(root, base_dir).startswith("."):
            continue
        if not os.path.relpath(root, base_dir).startswith("BenchmarkHTC"):
            continue
        print(root)
        print(os.path.relpath(root, base_dir))
        if 'params.json' in files and '8_summary.json' in files:
            params_path = os.path.join(root, 'params.json')
            summary_path = os.path.join(root, '8_summary.json')
            

            with open(params_path, 'r') as f:
                params = json.load(f)
                params["file_path"] = os.path.relpath(root, base_dir)

            with open(summary_path, 'r') as f:
                summary = json.load(f)

            final_mccs = summary.get('final_mccs', {})
            row = {**params, **final_mccs}
            data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    base_dir = '/home/ilmari/python/limited-angle-tomography'
    df = read_params_and_summary(base_dir)
    df["sum_mcc"] = df["a"] + df["b"] + df["c"] + df["d"]

    # If use_autoencoder_reg is 0, autoencoder_patch_size should be 0
    df.loc[df["use_autoencoder_reg"] == 0, "autoencoder_patch_size"] = 0
    # if 'synth' in ae path and use_autoencoder_reg != 0, skip
    df = df[~((df["autoencoder_path"].str.contains("synth")) & (df["use_autoencoder_reg"] != 0))]
    # if 'synth' in ae path skip
    #df = df[~(df["autoencoder_path"].str.contains("synth"))]
    
    # Changing cols: filter_raw_sinogram_with_a, use_tik_reg, use_tv_reg, time_limit, use_no_model, p_loss
    # Find the best 3 (max sum_mcc)
    best3 = df.sort_values("sum_mcc", ascending=False).head(5)
    print(best3)
    # To excel
    best3.to_excel("best.xlsx")
    #exit()
    
    # For each parameter plot a bar plot of the mean sum_mcc
    fig, ax = plt.subplots(3, 3, figsize=(20, 12))
    for i, col in enumerate(["filter_raw_sinogram_with_a",
                             "use_tik_reg",
                             "use_tv_reg",
                             "time_limit_s",
                             "use_no_model",
                             "p_loss",
                             "use_autoencoder_reg",
                             "autoencoder_patch_size"]):
        # Scale y values so largest is 1
        y = df.groupby(col)["sum_mcc"].mean()
        y = y - y.mean()
        y.plot(kind="bar", ax=ax[i//3, i%3])
        
        ax[i//3, i%3].set_title(f"Difference from mean MCC by {col}")
        ax[i//3, i%3].set_xlabel(col)
        ax[i//3, i%3].set_ylabel("Difference from mean MCC")
        
        # In each bar, add the mean value, and sample size
        for j, value in enumerate(y):
            vcs = df[col].value_counts()
            ax[i//3, i%3].text(j, value, f"{value:.2f}\n(n={vcs.iloc[j]})",
                               ha="center", va="bottom")
            
        
    plt.tight_layout()
    
    
    
    # For each parameter, find the best value
    for col in ["filter_raw_sinogram_with_a",
                "use_tik_reg",
                "use_tv_reg",
                "time_limit_s",
                "use_no_model",
                "p_loss",
                "use_autoencoder_reg",
                "autoencoder_patch_size"]:
        print(df.groupby(col)["sum_mcc"].mean())
        
    # Visualize the distribution of sum_mcc
    plt.figure(figsize=(10, 6))
    sns.histplot(df["sum_mcc"], kde=True)
    plt.title("Distribution of sum_mcc")
    plt.xlabel("sum_mcc")
    plt.ylabel("Frequency")

    # Pairplot to visualize relationships between parameters and sum_mcc
    sns.pairplot(df, vars=["filter_raw_sinogram_with_a", "use_tik_reg", "use_tv_reg", "time_limit_s", "use_no_model", "p_loss", "sum_mcc"])

    # Heatmap to visualize correlation between parameters and sum_mcc
    plt.figure(figsize=(12, 8))
    corr_matrix = df[["filter_raw_sinogram_with_a", "use_tik_reg", "use_tv_reg", "time_limit_s", "use_no_model", "p_loss", "sum_mcc"]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
        
    
    
    
    
        
    