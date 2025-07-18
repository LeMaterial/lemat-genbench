import glob
import pickle

import pandas as pd

if __name__ == "__main__":
    mlips = ["orb", "mace"]

    dir_name = "test_small_lematbulk"
    composition_path = "data/sample_lematbulk.pkl"
    full_df_dict = {}
    for mlip in mlips: 
        full_df = []
        df_paths = glob.glob(                    
                        "data/"
                        + dir_name
                        + "/"
                        + mlip
                        + "/*"
                        )
        for path in df_paths:
            with open(path, "rb") as f:
                temp_df = pickle.load(f)
            if len(full_df) == 0:
                full_df = temp_df
            else:
                full_df = pd.concat([full_df, temp_df])
        
        full_df_dict[mlip] = full_df
    
    with open(composition_path, "rb") as f:
        composition_df = pickle.load(f)
    print(len(composition_df))
    for mlip in mlips:
        print(len(composition_df))
        for column in full_df_dict[mlip].columns:
            full_df_dict[mlip].rename(columns={column: mlip.capitalize()+column}, inplace=True)
        composition_df = pd.concat([composition_df.reset_index(drop=True),
                                    full_df_dict[mlip].reset_index(drop=True)], axis=1)
    
    composition_df.to_pickle("data/full_reference_df.pkl")