import glob

import pandas as pd
from pandarallel import pandarallel
from pymatgen.core import Structure
from smact.screening import smact_validity


def initialize_parallel_processing():
    """
    Initialize parallel processing using pandarallel.

    This function sets up pandarallel for parallel processing of pandas operations,
    enabling a progress bar for better visibility of the computation progress.
    """
    pandarallel.initialize(progress_bar=True)


# Set up parallel processing
initialize_parallel_processing()

# The imported modules and functions are used as follows:
# - smact_validity: To check the charge neutrality of compositions
# - matplotlib.pyplot: For creating plots and visualizations
# - pandas: For data manipulation and analysis
# - pandarallel: To enable parallel processing of pandas operations


# Load the data
# Load the data into a dataframe

data = glob.glob("data/lematbulk_cifs/*")
comp = []
for cif in data:
    structure = Structure.from_file(cif)
    form = structure.formula
    form_no_space = form.replace(" ", "")
    comp.append(form_no_space)

df = pd.DataFrame(comp, columns=["Composition"])
# print(df)


# Run the SMACT validity test on the GNoME materials
df["smact_valid"] = df["Composition"].parallel_apply(
    smact_validity, **{"oxidation_states_set": "smact14"}
)  # Alloys will pass the test
counts = [len(df), df["smact_valid"].sum()]
print(counts)
print(counts[1]/counts[0])