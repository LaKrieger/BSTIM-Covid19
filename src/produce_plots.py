from plot_curves_window import curves as curves_window
from plot_curves_window_trend import curves as curves_window_trend
from plot_window_germany import curves as germany_map
from plot_interaction_kernel_window import interaction_kernel
from shared_utils import make_county_dict
import os
import subprocess
import concurrent.futures
import pandas as pd
import shutil

workers = 128

def main():

    def plot_curves(c):
        curves_window(start, c, n_weeks=3, model_i=35, save_plot=True)
        curves_window_trend(start, c, save_plot=True)
        return c

    start = int(os.environ["SGE_DATE_ID"]) 
    county_dict = make_county_dict()
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    year = str(start_day)[:4]
    month = str(start_day)[5:7]
    day = str(start_day)[8:10]
    figures_path = "/p/project/covid19dynstat/autostart/BSTIM-Covid19_Window_Final/figures/{}_{}_{}/".format(year, month, day)
    shared_path = "/p/project/covid19dynstat/shared_assets/figures/{}_{}_{}/".format(year, month, day) 

    if os.path.isfile(os.path.join(figures_path, 'metadata.csv')):
        os.remove(os.path.join(figures_path, 'metadata.csv'))

    germany_map(start, save_csv=True)
    interaction_kernel(start, save_plot=True)
    
    completed_counties = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for c in executor.map(plot_curves, county_dict.keys()):
            completed_counties.append(c)

    metadata_date_path = os.path.join(figures_path, "metadata.csv")
    metadata_total_path = "/p/project/covid19dynstat/shared_assets/metadata.csv"
    metadata_date_df = pd.read_csv(metadata_date_path)
    metadata_total_df = pd.read_csv(metadata_total_path)
    metadata_date_df_sorted = metadata_total_df.copy()
    probText_sorted = []
    for key in metadata_total_df["countyID"]:
        probText_sorted.append(float(list(metadata_date_df[metadata_date_df["countyID"]==key]["probText"])[0]))
    metadata_date_df_sorted["probText"] = probText_sorted
    metadata_date_df_sorted.to_csv(metadata_date_path, index=False)

    cwdir = r'.'
    # crop the images
    crop_command = r"find {} -type f -name '*.png' -exec convert {} -trim {} \;".format(figures_path, "{}", "{}") 
    returnval = subprocess.run(crop_command, check=False, shell=True, cwd=cwdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(returnval.stdout.decode("ascii"))
    print(returnval.stderr.decode("ascii"))
    shutil.rmtree(shared_path)
    shutil.copytree(figures_path, shared_path)

                             
if __name__ == '__main__':
    main()