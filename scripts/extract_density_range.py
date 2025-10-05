from __future__ import annotations

import pandas as pd 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    density_df = pd.read_csv("/home/samue/lemat-genbench/data/lematbulk_density_properties.csv")
    plt.hist(density_df["Density(atoms/A^3)"])
    plt.savefig("atomic_density_hist.png")
    plt.clf()
    plt.hist(density_df["Density(g/cm^3)"])
    plt.savefig("mass_density_hist.png")
    plt.clf()
    print(max(density_df["Density(atoms/A^3)"]))
    print(min(density_df["Density(atoms/A^3)"]))