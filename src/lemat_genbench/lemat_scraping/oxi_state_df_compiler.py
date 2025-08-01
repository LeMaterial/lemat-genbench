import glob
import pickle

import pandas as pd

if __name__ == "__main__":

    dir_name = "lematbulk_oxi_data"
    composition_path = "data/lematbulk_oxi_no_strut.pkl"
    
    full_df = []
    df_paths = glob.glob(                    
                    "data/"
                    + dir_name
                    + "/*"
                    )
    for path in df_paths:
        print(path)
        with open(path, "rb") as f:
            temp_df = pickle.load(f)
        if len(full_df) == 0:
            full_df = temp_df
        else:
            full_df = pd.concat([full_df, temp_df])
        
    full_df.reset_index(inplace = True)
    
    full_df.to_pickle("data/test.pkl")