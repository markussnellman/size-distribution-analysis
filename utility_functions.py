import numpy as np
import base64
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import csv
from io import BytesIO, StringIO
from data_analysis import fit_curve, gaussian, reconstruct_gaussian
from nptdms import TdmsFile
from dash import html
from itertools import cycle


# Functions

# Todo: fix handling multiple traces
def read_SMPS(lines):
    """
    Reads lines from SMPS.txt file from SMPS system and returns diameters and dNdlogDp.
    """
    dNdlogDp = []
    diameter = []

    reader = csv.reader(lines, delimiter='\t')
    first_row = []
    start_idx = 0
    end_idx = 0

    # skip 25 rows
    for _ in range(25):
        next(reader)

    for i, row in enumerate(reader):
        if i == 0:
            first_row = row
        elif i == 1:
            lst = row[8:-29]
            for j, s in enumerate(lst):
                if s:
                    start_idx = j + 8
                    break

            for j in range(len(lst) - 1, -1, -1):
                if lst[j]:
                    end_idx = j + 8
                    break
            
            diameter = [float(s) for s in first_row[start_idx:end_idx+1]]
            try:
                dNdlogDp.append([float(s) for s in row[start_idx:end_idx+1]])
            except ValueError:
                print(f"Value error at row index {i}")
            
        else:
            try:
                dNdlogDp.append([float(s) for s in row[start_idx:end_idx+1]])
            except ValueError:
                print(f"Value error at row index {i}")


    if len(dNdlogDp) == 1:
        return diameter, dNdlogDp[0]
    else:
        return diameter, dNdlogDp


def parse_content(content, filename):
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try: 
        if 'tdms' in filename:
            # Assume that the user uploaded a TDMS file
            # Assuming TDMS file content is in binary format
            tdms_file = TdmsFile(BytesIO(decoded))

            # Convert to dataframe
            raw_df=tdms_file.as_dataframe()

            if raw_df.empty:
                df = raw_df
                div = html.Div([f'{filename} is empty!'])
            else:
                df = raw_df.iloc[:, [-2, -1]]
                df.columns = ['DMA2', 'Conc.']
                div = html.Div([f'Uploaded {filename}.'])

        elif 'txt' in filename:
            decoded_text = decoded.decode('utf-8')
            lines = decoded_text.split('\n')
            # Here we need to check if the txt is from our setup or TSI
            # Lines is a list of strings corresponding to each line
            # TSI files have first line called Sample File
            if ("Sample File" in lines[0]):
                diameter, dNdlogDp = read_SMPS(lines)

                # dNdlogDp can be a list of lists
                # Then should return lists of dfs
                if isinstance(dNdlogDp[0], list):
                    # df = pd.DataFrame({'Diameter': diameter, **{f'dNdlog(Dp) Trace{i+1}': dNdlogDp[i] for i in range(len(dNdlogDp)) if len(dNdlogDp[i]) > 0}})
                    df = [pd.DataFrame({'Diameter': diameter, f'dNdlog(Dp) Trace{i+1}': dNdlogDp[i]}) for i in range(len(dNdlogDp)) if len(dNdlogDp[i]) > 0]
                else:
                    df = pd.DataFrame({'Diameter': diameter, 'dNdlog(Dp)': dNdlogDp})
                

            # Else assume it's from our setup
            else:
                df = pd.read_csv(StringIO(decoded_text), delimiter='\t', header=0, usecols=[6, 7], names=['DMA2', 'Conc.'])

            div = html.Div([f'Uploaded {filename}.'])

        else:
            df = None
            div = html.Div(['File type needs to be TDMS or an SMPS txt.'])
    
    except Exception as e:
        print(e)
        df = None
        div = html.Div(['There was an error processing this file.'])

    return div, df


def generate_plot(plotted_dataset_list, dataframes, logstate):
    # Create initial empty figure
    fig = px.scatter()

    # Loop over plotted datasets
    for to_plot in plotted_dataset_list:
        # Because the dataframes are stored as JSON, they need to be read as JSON
        df = pd.read_json(dataframes[to_plot], orient='split')
        # If it's a best fit dataframe to plot, we need to iterate over all y columns
        # which are the columns with index > 0
        if ("_best_fit" in to_plot):
            colors = cycle(['red', 'palegreen', 'mediumturquoise', 'mediumslateblue', 'darkgreen'])
            # y_cols = [col for col in df.columns if col != 'DMA2']
            y_cols = [df.columns[i] for i in range(len(df.columns)) if i > 0]
            for col in y_cols: 
                trace = px.line(data_frame=df, x=df.columns[0], y=col, color_discrete_sequence=[next(colors)]).update_traces(name=to_plot[:-5]).data[0]

                fig.add_trace(trace)
        else:
            trace = px.scatter(data_frame=df, x=df.columns[0], y=[df.columns[1]], color_discrete_sequence=['blue']).update_traces(name=to_plot[:-5]).data[0]

            fig.add_trace(trace)

    if 'Log X' in logstate:
        fig = fig.update_layout(
            xaxis=dict(type='log')
        )
    if 'Log Y' in logstate:
        fig = fig.update_layout(
            yaxis=dict(type='log')
        )
    if 'Log X' not in logstate:
        fig = fig.update_layout(
            xaxis=dict(type='linear')
        )
    if 'Log Y' not in logstate:
        fig = fig.update_layout(
            yaxis=dict(type='linear')
        )

    fig.update_layout(xaxis_title='Mobility diameter (nm)',
                      yaxis_title='Concentration (cm^-3)',)
    
    # Return updated figure
    return fig


def popt_to_str(popt):
    # Convert optimal parameters to a string representation
    popt_str = ''
    for key, val in zip(popt.keys(), popt.values()):
        popt_str += key + ": " + str(round(val, 2)) + " "
    return popt_str


def best_fit_to_df(x, best_fit_params):
    best_fits = np.sum(reconstruct_gaussian(x, *np.abs(best_fit_params)), axis=1)
    df = pd.DataFrame({'DMA2': x, 'Conc.': best_fits})
    return df


def save_data():
    pass


if __name__=="__main__":
    # Testing
    file_path = 'SMPS.txt'

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # Adjust delimiter

        # Skip the first 25 rows
        for i in range(25):
            next(reader)

        diameter_row = next(reader)
        concentration_row = next(reader)

        # We need to find start and end index by traversing diameter row until we come
        # across element != ''
        start_idx = None
        end_idx = None
        for i in range(8, len(concentration_row)):
            if concentration_row[i] != '':
                start_idx = i   
                break
        
        for i in range(len(concentration_row) - 30, 0, -1):
            if concentration_row[i] != '':
                end_idx = i + 1
                break

        diameter = diameter_row[start_idx:end_idx]
        concentration = concentration_row[start_idx:end_idx]

        print(diameter, concentration)