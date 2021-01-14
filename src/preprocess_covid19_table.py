import pandas as pd
import numpy as np
import json
import csv
import argparse
import re
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0]  # Repeat as needed
    grand_parent = os.path.split(parent_directory)[0]
    file = grand_parent+"/LKOS_Daten_COVID19.csv"
    data = pd.read_csv(file, encoding='latin-1')

    # change names to ids and drop natural names (but save them for later)
    data["ID"]= list(range(len(data.index)))
    data.set_index("ID", drop = True, inplace = True)
    id_to_name = data[["Bestätigte (Neu-)Fälle pro Tag"]]
    id_to_name.rename(columns={"Bestätigte (Neu-)Fälle pro Tag": "NL Name"}, inplace = True)
    data.drop("Bestätigte (Neu-)Fälle pro Tag", axis = 1, inplace = True)

    data.columns = pd.to_datetime(data.columns, dayfirst=True)
    df = data.transpose()

    df.to_csv(grand_parent+"/preprocessedLKOS.csv")
    id_to_name.to_csv(grand_parent + "/ID_to_name.csv")


