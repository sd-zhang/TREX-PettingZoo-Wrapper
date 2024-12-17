# we're going to import a bunch of csv files from _data/, which are exports from a tensorboard log
# the files can go into a pandas DF, and then we can
# process them into a single csv file
# that can be used to plot the data

# the csvs are named as follows:
# Max_summary [Metric].csv are the metrics that follow the values occuring at the maximum return
# Power_quality [Metric].csv are the metrics that follow the values in general
# metrics are:
#   Daily_energy_exported
#   Daily_energy_imported
#   Daily_peak_export
#   Daily_peak_import
#   Daily_ramping_rate
#   Daily_load_factor
#   Monthly_load_factor
#   Max_Return
#   Total_peak_export
#   Total_peak_import


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_power_metric_csvs():
    # get the list of files
    files = os.listdir('_data/Power Metrics')
    print(files)
    print(len(files))

    # create a list of dataframes
    dfs = []
    for file in files:
        # print(file)
        df = pd.read_csv('_data/Power Metrics/' + file)
        # print(df)
        dfs.append(df)

    # print(dfs)
    # delete the wall time column
    for df in dfs:
        del df['Wall time']

    # rename the metrics to differentiate between the Max_summary and Power_quality
    # names can be found in files
    metrics = []
    for df, filename in zip(dfs, files):
        # at this point, the df should contain two colums: step and value
        print(df.columns)

        # look for a substring in the filename to determine pre processing
        power_metrics = ['export', 'import', 'ramping', 'peak', 'valley']
        if any(power_metric in filename for power_metric in power_metrics):
            df['Value'] = df['Value']/1000

        ramping = ['ramping']
        if any(ramp in filename for ramp in ramping):
            df['Value'] = df['Value']/24

        daily_export = ['daily_export']
        if any(daily in filename for daily in daily_export):
            df['Value'] = df['Value']/1.7



        if 'Max_summary' in filename:
            # we want to separate the suffix from the metric name and add 'Max_Return' as prefix to the metric name
            metric_name = filename.split('Max_summary ')[1][:-4]
            # replace the _ with a space
            metric_name = metric_name.replace('_', ' ')
            df.rename(columns={'Value': 'Max_Return ' + metric_name}, inplace=True)
        elif 'Power_quality' in filename:
            # we want to separate the suffix from the metric name and add 'Power_quality' as prefix to the metric name
            metric_name = filename.split('Power_quality ')[1][:-4]
            metric_name = metric_name.replace('_', ' ')
            metrics.append(metric_name)
            df.rename(columns={'Value': metric_name}, inplace=True)
        else:
            print('error, filename not recognized')
            sys.exit(1)
        print('into', df.columns)
        print('.')


    # merge the dataframes based on the step column
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on='Step', how='outer')

    # turn the step column into the index
    df_merged.set_index('Step', inplace=True)
    print('returning df with the following columns:')
    print(df_merged.columns)

    return df_merged, metrics

def process_rl_metrics_csvs():
    # loads and processes the RL metrics
    # get the list of files
    files = os.listdir('_data/RL Metrics')
    print(files)
    print(len(files))
    metrics = []

    # create a list of dataframes
    dfs = []
    for file in files:
        # print(file)
        df = pd.read_csv('_data/RL Metrics/' + file)
        # print(df)
        dfs.append(df)

    # print(dfs)
    # delete the wall time column
    for df in dfs:
        del df['Wall time']

    for df, filename in zip(dfs, files):
        # at this point, the df should contain two colums: step and value
        print(df.columns)
        metric_name = filename[:-4]
        # look for a substring in the filename to determine pre processing
        actor_loss = ['Actor Loss']

        df.rename(columns={'Value': metric_name}, inplace=True)
        metrics.append(metric_name)



    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on='Step', how='outer')

    # chop high outliers above 99.9999 percentile for actor loss
    df_merged = df_merged[df_merged['Actor Loss'] < df_merged['Actor Loss'].quantile(0.999999)]

    # turn the step column into the index
    df_merged.set_index('Step', inplace=True)

    return df_merged, metrics

def plot_power_metrics(df, metrics):
    # the df will contain a step column, and for every metric a 'Max_Return' column and a prefixless column
    # we want to plot all the metricw with the same name into the same field
    # all metrics will use step as the x axis
    # we want the prefixless column with a lower alpha than the Max_Return column, but the same color
    # we don't necessarily care about the order of the metrics

    num_metrics = len(metrics) -1
    num_rows = int(np.ceil(num_metrics/3))
    num_cols = int(np.ceil(num_metrics/num_rows))
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)
    color = 'blue'
    for index, metric in enumerate(metrics):
        if metric != 'Return':
            # plot the metric, plot the max_return and rare metric
            # plot the prefixless column
            #extract the prefixless column and the step values for each entry
            rare_metric = df[metric]
            # extract index values
            rare_metric_index = rare_metric.index.values
            max_index = np.amax(rare_metric_index)
            min_index = np.amin(rare_metric_index)
            #rescale to 0-114
            rare_metric_index = (rare_metric_index - min_index) / (max_index - min_index) * 114
            # extract the actual values
            rare_metric = rare_metric.values
            plot_row = int(index/num_cols)
            plot_col = index % num_cols
            ax[plot_row, plot_col].scatter(rare_metric_index, rare_metric, color=color, alpha=0.1)

            max_return = df['Max_Return ' + metric].values
            max_return_index = df['Max_Return ' + metric].index.values
            max_return_index = (max_return_index - min_index) / (max_index - min_index) * 114
            ax[plot_row, plot_col].plot(max_return_index, max_return, color=color)
            if plot_row == num_rows - 1:
                ax[plot_row, plot_col].set_xlabel('Episode')
            # if 'export' in metric or 'import' in metric or 'ramping' in metric: # convert from W to kW by dividing by 1000if ['export', 'import', 'ramping'] in metric:
            #    ax[plot_row, plot_col].set_ylabel('kW')
            ax[plot_row, plot_col].set_title(metric)

    fig.set_size_inches(10, 10)

    plt.legend()
    plt.show()
    plt.close()

    #figure size

    # for all rare metrics, calculate the corellation with all other rare metrics
    # print the correlation matrix
    for index, metric in enumerate(metrics):
        correlation = df['Return'].corr(df[metric])
        max_returncorrelation = df['Max_Return ' +'Return'].corr(df['Max_Return ' + metric])
        print(f'corr between Return and {metric}: {correlation}, {max_returncorrelation}')




def plot_RL_metrics(df, metrics):
    # the df will contain a step column, and for every metric a 'Max_Return' column and a prefixless column
    # we want to plot all the metricw with the same name into the same field
    # all metrics will use step as the x axis
    # we want the prefixless column with a lower alpha than the Max_Return column, but the same color
    # we don't necessarily care about the order of the metrics



    num_metrics = len(metrics)
    num_rows = int(np.ceil(num_metrics/2))
    num_cols = int(np.ceil(num_metrics/num_rows))
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, figsize=(15, 17))
    color = 'blue'
    for index, metric in enumerate(metrics):
        # plot the metric, plot the max_return and rare metric
        # plot the prefixless column
        #extract the prefixless column and the step values for each entry
        rare_metric = df[metric]
        # extract index values
        # extract the actual values
        plot_row = int(index/num_cols)
        plot_col = index % num_cols
        # scatter the true values
        ax[plot_row, plot_col].plot(rare_metric, color=color, alpha=0.2)

        smoothed_rare_metric = rare_metric.rolling(10).mean()
        ax[plot_row, plot_col].plot(smoothed_rare_metric, color=color)
        ax[plot_row, plot_col].set_title(metric)
        if plot_row == num_rows - 1:
            ax[plot_row, plot_col].set_xlabel('Episode')


    plt.legend()
    plt.show()

    plt.close()

def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    means = []
    std = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        means.append(np.nanmean(temp))
        std.append(np.nanstd(temp))
    return means, std

def plot_Return():
    # plot the return metric separately

    df1 = pd.read_csv('_data/Return/run1_Building_mean_returns.csv')
    df2 = pd.read_csv('_data/Return/run2_Building_mean_returns.csv')
    df3 = pd.read_csv('_data/Return/run3_Building_mean_returns.csv')

    fig, ax = plt.subplots(1, 1)
    color = 'blue'
    run1_return_metric = df1['Value']
    run1_return_metric_index = run1_return_metric.index.values #were gonna plot on this axis
    run1_return_metric = run1_return_metric.values.tolist()

    run2_return_metric = df2['Value']
    run2_return_metric_index = run2_return_metric.index.values
    run2_return_metric = run2_return_metric.values.tolist()

    run3_return_metric = df3['Value']
    run3_return_metric_index = run3_return_metric.index.values
    run3_return_metric = run3_return_metric.values.tolist()

    return_mean, return_std = avgNestedLists([run1_return_metric, run2_return_metric, run3_return_metric])

    return_mean = np.array(return_mean[:110])
    return_std = np.maximum(1.0, np.array(return_std[:110]))
    indices= np.arange(len(return_mean))
    #calculate the mean return
    ax.plot(indices, return_mean, color=color, alpha=1, label='ALEX RL')
    ax.fill_between(indices, return_mean - return_std, return_mean + return_std, color=color, alpha=0.3)

    #add a straight line at height 144.467 with label ALEX DP
    ax.axhline(y=144.467, color='r', linestyle=':', label='ALEX DP')

    # #add a straight line at height 127.993 with label IndividualDERMS
    # ax.axhline(y=127.993, color='g', linestyle=':', label='IndividualDERMS')

    # add line at height 61.065 with label NoDERMS in color orange
    # orange_color = (1.0, 0.647, 0.0)
    # ax.axhline(y=61.065, color=orange_color, linestyle=':', label='NoDERMS')



    fig.set_size_inches(5, 4)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Bill Savings [$]')
    ax.set_title('Average Building Bill Savings')

    #make figsize smaller

    plt.legend()
    plt.show()

    plt.close()

if "__main__" == __name__:
    df, metrics = process_power_metric_csvs()
    df_columns = df.columns.tolist()
    max_return_metrics = [metric for metric in df.columns if 'Max_Return' in metric]
    print('Max return metric summary:')
    for max_return_metric in max_return_metrics:
        df_max_return_metric = df[max_return_metric]
        # get last entry that is not nan
        last_valid_entry = df_max_return_metric[df_max_return_metric.notnull()].iloc[-1]
        print(f'{max_return_metric}: {last_valid_entry}')

    plot_power_metrics(df, metrics)

    df, metrics = process_rl_metrics_csvs()
    df_columns = df.columns.tolist()
    plot_RL_metrics(df, metrics)
    print(metrics)

    plot_Return()





