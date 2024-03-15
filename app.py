from dash import Dash, dcc, Output, Input, State, html, no_update, ctx, dash_table
import dash_bootstrap_components as dbc   
import numpy as np
import pandas as pd
from utility_functions import parse_content, generate_plot, popt_to_str
from data_analysis import fit_complex_model, build_complex_model
import plotly.graph_objects as go

data = {}

# Components
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL], prevent_initial_callbacks="initial_duplicate")
server = app.server
title = dcc.Markdown(children='# Size distribution analysis')
main_graph = dcc.Graph(figure={})
upload = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-top': '10px',
            'margin-bottom': '10px'
        })
status_div = html.Div(id='output-data-upload')

# Table
# Placeholder data
df = pd.DataFrame({'Tot. conc. (cm^-3)': [0], 'Surf. area conc. (nm^2/cm^3)': [0]})
table = dash_table.DataTable(
    data = df.to_dict('records'),
    columns = [{'name': i, 'id': i} for i in df.columns],
    id = 'table',)

file_dropdown = dcc.Dropdown({}, [], id='file-dropdown', style={'margin-bottom': '10px'})
subfile_dropdown = dcc.Dropdown({}, [], id='subfile-dropdown', style={'margin-bottom': '10px'})
instructions_modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Curve fit instructions")),
                dbc.ModalBody("The curve fit can fit an arbitrary number of modes and accepts comma separated parameters on the form\nfunction_1, param_1, ..., param_2, function_2...\n\nThe currently supported functions are normal and lognormal which both takes params amplitude, center (GMD for lognormal), and sigma (GSD for lognormal). The better the guess, the better the fit. Note that GSD needs to be strictly larger than 1.\n\nExample input: normal, 10000, 20, 1, log, 1e7, 30, 1.5."),
                dbc.ModalFooter(
                    html.Button("Close", id="close-modal", n_clicks=0)
                ),
            ],
            id="instructions-modal",
            is_open=False,
            size='xl',
        )
fit_params_input = dcc.Input(id='fit-params', type='text', placeholder='CS input pars', style={'margin-left': '20px'})
curve_fit_dropdown = dcc.Dropdown({}, [], id='curve-fit-dropdown', style={'margin-bottom': '10px'})

# Layout
app.layout = dbc.Container([title, 
                            dcc.Checklist(['Log X', 'Log Y'], inline=True, id='log-check'),
                            main_graph, 
                            html.Button("Remove plots", id="remove-plots", n_clicks=0),
                            upload, 
                            status_div, 
                            html.H3("Data"),
                            table,
                            html.H3("Select Dataset:"), 
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            file_dropdown, 
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            subfile_dropdown,
                                        ]
                                    ),
                                ]
                                ),
                            html.Button('Plot/Remove plot', id='plot-data-btn', n_clicks=0, style={'margin-right': '10px'}), 
                            html.Button('Curve fit instructions', id='instructions-btn', style={'margin-right': '0px'}),
                            instructions_modal,
                            fit_params_input,
                            html.Button('Curve fit', id='curve-fit-btn', n_clicks=0), 
                            html.H3("Best fit curves"), 
                            curve_fit_dropdown, 
                            html.Button('Plot/Remove plot', id='plot-fit-btn', n_clicks=0, style={'margin-right': '10px'}), 
                            html.Button("Save data", id='save-btn', n_clicks=0),
                            dcc.Download(id="download"), # Invisible Save Link
                            dcc.Store(id='data-store', data={'plotted': [], 'dataframes': {}, 'best_fit_params': {}})])

# Callbacks
@app.callback(Output('file-dropdown', 'options'),
            Input('upload-data', 'filename'),
            Input('file-dropdown', 'options'),
            prevent_initial_call=True)
def update_dropdown(filename, options):
    if 'tdms' in filename:
        options[filename] = filename
        return options
    if 'txt' in filename or 'SMPS' in filename:
        options[filename] = filename
        return options
    else:
        return no_update
    

@app.callback(Output('subfile-dropdown', 'options'),
              Output('subfile-dropdown', 'value'),
              Input('file-dropdown', 'value'),
              State('data-store', 'data'),
              prevent_inital_call=True)
def update_sub_dropdown(filename, data):
    """
    This is used to handle SMPS files that can have multiple traces.
    """
    if isinstance(filename, str):
        if 'SMPS' in filename:
            options = []
            df_names= data['dataframes'].keys()
            for name in df_names:
                if filename in name:
                    options.append({'label': name, 'value': name})
            return options, options[0]['value']
        return [], None
    else:
        return no_update, no_update


# To fix: handling of other files than TDMS
@app.callback(Output('output-data-upload', 'children'),
              Output('data-store', 'data', allow_duplicate=True),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('data-store', 'data'),
              prevent_initial_call=True)
def update_output(content, name, data):
    if content is not None:
        child, df = parse_content(content, name)
        dataframes = data['dataframes']
        # Dataframe needs to be converted to Json
        # Here we need to figure out how to handle multiple traces
        if isinstance(df, list):
            for i in range(len(df)):
                dataframes[name + " " + f"Trace {i + 1}"] = df[i].to_json(date_format='iso', orient='split')
        else:
            dataframes[name] = df.to_json(date_format='iso', orient='split')
        data['dataframes'] = dataframes
        return child, data


# To fix: handling of other files than TDMS
@app.callback(Output('table', 'data'),
              Input('file-dropdown', 'value'),
              State('data-store', 'data'),
              prevent_inital_call=True)
def update_table_from_dataset_dropdown(value, data):
    """
    Computes total concentration and surface area concentration.
    """
    if isinstance(value, str):
        if 'SMPS' not in value:
            df = pd.read_json(data['dataframes'][value], orient='split')

            x = df[df.columns[0]] 
            y = df[df.columns[1]]
        
            # Compute total concentration
            conc = np.trapz(y, x)

            # Compute surface area concentration
            sa_conc = np.sum([d ** 2 * np.pi * num for d, num in zip(x, y)])

            df = pd.DataFrame({'Tot. conc. (cm^-3)': [conc], 'Surf. area conc. (nm^2/cm^3)': [sa_conc]})
            return df.to_dict('records')
        else:
            df = pd.DataFrame({'Tot. conc. (cm^-3)': [0], 'Surf. area conc. (nm^2/cm^3)': [0]})
            return df.to_dict('records')
    return no_update


@app.callback(Output('table', 'data', allow_duplicate=True),
              Input('file-dropdown', 'value'),
              Input('subfile-dropdown', 'value'),
              State('data-store', 'data'),
              prevent_inital_call=True)
def update_table_from_subfile_dropdown(value, subvalue, data):
    """
    Computes total concentration and surface area concentration.
    """
    if isinstance(value, str) and isinstance(subvalue, str):
        
        df = pd.read_json(data['dataframes'][subvalue], orient='split')

        x = df[df.columns[0]] 
        y = df[df.columns[1]] / 64
    
        # Compute total concentration
        conc = np.trapz(y, x)

        # Compute surface area concentration
        sa_conc = np.sum([d ** 2 * np.pi * num for d, num in zip(x, y)])

        df = pd.DataFrame({'Tot. conc. (cm^-3)': [conc], 'Surf. area conc. (nm^2/cm^3)': [sa_conc]})
        return df.to_dict('records')

    return no_update


@app.callback(Output(main_graph, component_property='figure', allow_duplicate=True),
              Output('data-store', 'data', allow_duplicate=True),
              Input('file-dropdown', 'value'),
              Input('plot-data-btn', 'n_clicks'),
              State('data-store', 'data'),
              State('subfile-dropdown', 'value'),
              State('log-check', 'value'),
              prevent_initial_call=True)
def update_graph_on_click(selected_dataset, clicks, data, trace, logstate):
    
    # Check if button clicked and dataset is selected
    if isinstance(selected_dataset, str) and clicks > 0 and 'plot-data-btn' == ctx.triggered_id:

        # if "SMPS" not in selected_dataset:
        print(trace) # Check what trace looks like when we don't have an SMPS file
        # Since it's an empty list, we can shorten this function
        # But now, there's a bug when we switch from a SMPS file to a regular file: the SMPS file trace name persists
        if isinstance(trace, str):
            selected_dataset = trace
        
        plotted_datasets_list = data['plotted']

        # Check if the selected dataset is already plotted
        if selected_dataset in plotted_datasets_list:
            # Remove the dataset from the list
            plotted_datasets_list.remove(selected_dataset)

        else:
            # Add the dataset to the list
            plotted_datasets_list.append(selected_dataset)

        # Update the plotted datasets in the Store
        data['plotted'] = plotted_datasets_list

        # Generate updated plot based on list of plotted data
        if logstate is not None: 
            updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], logstate)
        else:
            updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], [])

        return updated_figure, data
    
    # If no update needed, return current figure and plotted dataset
    return no_update, data


@app.callback(Output(main_graph, component_property='figure', allow_duplicate=True),
              Output('data-store', 'data'),
              Input('remove-plots', 'n_clicks'),
              State('data-store', 'data'),
              State('log-check', 'value'),
              prevent_inital_call=True)
def on_click_remove_plots(clicks, data, logstate):
    if clicks > 0 and 'remove-plots' == ctx.triggered_id:
        plotted_datasets_list = []
        data['plotted'] = plotted_datasets_list
        if logstate is not None: 
            updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], logstate)
        else:
            updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], [])
        return updated_figure, data
    else:
        return no_update, no_update


@app.callback(
        Output(main_graph, component_property='figure', allow_duplicate=True),
        Input('log-check', 'value'),
        State(main_graph, component_property='figure'),
        prevent_inital_call=True,
)
def log_check(values, figure):
    if values is not None: 
        fig = go.Figure(figure)
        if 'Log X' in values:
            fig = fig.update_layout(
                xaxis=dict(type='log')
            )
        if 'Log Y' in values:
            fig = fig.update_layout(
                yaxis=dict(type='log')
            )
        if 'Log X' not in values:
            fig = fig.update_layout(
                xaxis=dict(type='linear')
            )
        if 'Log Y' not in values:
            fig = fig.update_layout(
                yaxis=dict(type='linear')
            )
        return fig
    else:
        return no_update
    
# html.Div(['File type needs to be TDMS or an SMPS txt.'])
# 
@app.callback(Output('data-store', 'data', allow_duplicate=True),
              Output('fit-params', 'value'),
              Output('output-data-upload', 'children', allow_duplicate=True),
              Input('curve-fit-btn', 'n_clicks'),
              Input('file-dropdown', 'value'),
              State('subfile-dropdown', 'value'),
              State('data-store', 'data'),
              Input('fit-params', 'value'),
              prevent_inital_call=True)
def on_click_fit_curve(n_clicks, filename, trace, data, params):
    # Check input is ok
    if isinstance(filename, str) and n_clicks > 0 and 'curve-fit-btn' == ctx.triggered_id and params is not None:
        
        if len(params.split(',')) % 4 == 0:
            print(f"Fitting curve with {params}")
            model, status = build_complex_model(params)
            print(status)

            if status == "Success":
                # Load the dataframe and extract x, y data
                dataframes = data['dataframes']

                if "SMPS" in filename and isinstance(trace, str):
                    filename = trace
                
                if "SMPS" in filename and isinstance(trace, str) is False:
                    # This is if the user has forgotten to choose a trace, if there are traces
                    return no_update, no_update

                df = pd.read_json(dataframes[filename], orient='split')
                # 0 th column = diameter, 1 st column = concentration
                x = np.array(df[df.columns[0]])
                y = np.array(df[df.columns[1]])

                # This is to handle exception if the model could not be fit, which can be the case
                # for difficult data and many assumed modes
                try:
                    # Fit the model
                    best_fit, components, popt = fit_complex_model(x, y, model)
                    
                    # Add best fit parameters to data store
                    data['best_fit_params'][filename] = popt

                    # Convert best fit + components to dataframe and add to data store 
                    columns = {df.columns[0] : x, 'Best fit': best_fit}
                    # If there are 2 or more functions making up the fit
                    # add them
                    if len(components.keys()) > 1:
                        columns.update(components)
                    df_components = pd.DataFrame(columns)
                    data['dataframes'][filename + '_best_fit'] = df_components.to_json(date_format='iso', orient='split')

                    # Return data and optimal parameters    
                    return data, "", html.Div(['Succesfully fitted model.'])
                
                except Exception as e:
                    return no_update, no_update, html.Div([str(e)])
            
            else:
                return no_update, no_update, html.Div([status])
            
        return no_update, no_update, html.Div(['Wrong input shape.'])
        
    return no_update, no_update, no_update


@app.callback(
    Output("instructions-modal", "is_open"),
    [Input("instructions-btn", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("instructions-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(Output('curve-fit-dropdown', 'options'),
              Input('file-dropdown', 'value'),
              State('subfile-dropdown', 'value'),
              Input('data-store', 'data'), # Changed from Input -> State
              prevent_initial_call=True)
def update_best_fit_dropdown(filename, trace, data):
    # Check that a file is selected
    if isinstance(filename, str):
        # Check that filename is not ""
        if filename != "":
            best_fit_params = data['best_fit_params']
            if "SMPS" in filename and isinstance(trace, str):
                filename = trace
            if "SMPS" in filename and isinstance(trace, str) is False:
                return no_update
            if filename in best_fit_params.keys():
                print("Updating best fit dropdown")
                popt = best_fit_params[filename]
                popt_string = popt_to_str(popt)
                return [{'label': popt_string, 'value': popt_string}]
            return [{'label': '', 'value': ''}]
        return no_update
    return no_update


@app.callback(Output(main_graph, component_property='figure'),
              Output('data-store', 'data', allow_duplicate=True),
              Input('plot-fit-btn', 'n_clicks'),
              Input('file-dropdown', 'value'),
              Input('subfile-dropdown', 'value'),
              State('data-store', 'data'),
              State('log-check', 'value'),)
def on_click_plot_best_fit(n_clicks, filename, trace, data, logstate):
    if 'plot-fit-btn' == ctx.triggered_id and filename:
        best_fit_params = data['best_fit_params']
        print("Plot best fit button clicked.")

        # Check that the selected file has associated best fit
        if "SMPS" in filename and isinstance(trace, str):
            filename = trace
        if "SMPS" in filename and isinstance(trace, str) is False:
            return no_update, no_update
        if filename in best_fit_params.keys():
            plotted_datasets_list = data['plotted']
            selected_dataset = filename + "_best_fit"

            # If best fit plotted, remove from plotted list
            if selected_dataset in plotted_datasets_list:
                plotted_datasets_list.remove(selected_dataset)

            # Else, add best fit to plotted list
            else:
                plotted_datasets_list.append(selected_dataset)

            # Update the plotted datasets in the Store
            data['plotted'] = plotted_datasets_list

            # Generate updated plot based on list of plotted data
            if logstate is not None:
                updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], logstate)
            else:
                updated_figure = generate_plot(plotted_datasets_list, data['dataframes'], [])

            return updated_figure, data

        return no_update, no_update
    
    else:
        return no_update, no_update


@app.callback(
        Output('download', 'data'),
        Input('save-btn', 'n_clicks'),
        State('file-dropdown', 'value'),
        State('subfile-dropdown', 'value'),
        State('data-store', 'data'),
        prevent_inital_call=True,
)
def on_click_save_data(_, filename, subfile, data):
    # Check input of
    if ctx.triggered_id == 'save-btn' and isinstance(filename, str):

        # This check is for if it's TSI file with traces
        if isinstance(subfile, str):
            filename = subfile
        
        # Get the df corresponding to selected file
        main_df = pd.read_json(data['dataframes'][filename], orient='split')

        # Check whether this file has a best fit, add to df
        best_fit_params = data['best_fit_params']
        # Save the best fit data
        if filename in best_fit_params.keys():
            df = pd.read_json(data['dataframes'][filename + "_best_fit"], orient='split')
            # The columns with idx 1 is the total best fit (column 0 is diameter/size)
            main_df['Best fit'] = df[df.columns[1]]
            # The other cols are the components
            if len(df.columns) >= 3:
                best_fit_cols = df.columns[2:]
                for col in best_fit_cols:
                    main_df[col] = df[col]

        return dcc.send_data_frame(main_df.to_csv, f"{filename[:-5]}.txt", index=False, sep='\t')
    return no_update


# Run app
if __name__=='__main__':
    app.run_server(debug=False) # Set to False when committing to GitHub
