# Aerosol size distribution analysis

This Dash app can be used to analyze size distribution data, including TSI SMPS files.

1. Drag and drop your file to the upload button.
2. Select the file you want from the dropdown. If it's a TSI SMPS file, you need to select the correct trace in the second dropdown.
3. To do curve fitting, comma separated values are used this way:

type_1, param_1, param_2, ... , param_n, type_2, param_1, param_2, ... param_n, ...

Supported types: "norm"/"normal" (Gaussian) and "log" (Lognormal)

Params: Both normal and log have amplitude, center (mean/geometric mean) and sigma (standard deviation/geometric standard deviation). These are provided as numbers. Note that GSD needs to be strictly larger than 1.

Example input (note the difference between log and normal): log, 2e7, 30, 1.4, norm, 5000000, 68, 2

4. If curve fitting is succesful, the best fit params are in the bottom dropdown. Fitting a new function will override them. They can be plotted like the regular data.
5. To save data including best fits, just select the dataset and click save.
