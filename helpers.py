from types import MappingProxyType

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.stats import ttest_ind, f_oneway
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import geopandas as gpd
from scipy.stats import shapiro

fonts = fm.findSystemFonts()

for font in fonts:
    fm.fontManager.addfont(font)

font_size = 11
font_family = 'Times New Roman'
# plot_style = 'seaborn-v0_8-paper'
plot_style = 'fivethirtyeight'

plt.style.use(plot_style)
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = font_family
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['figure.titlesize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 1
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['figure.constrained_layout.use'] = True

# ordinal_map = MappingProxyType({'Strongly Agree': 2,
#                'Agree': 1,
#                'Neutral': 0,
#                'Disagree': -1,
#                'Strongly Disagree': -2})

# ordinal_map = MappingProxyType({'Strongly Agree': 3,
#                                 'Agree': 3,
#                                 'Neutral': 2,
#                                 'Disagree': 1,
#                                 'Strongly Disagree': 1})

ordinal_map = MappingProxyType({'Strongly Agree': 5,
                                'Agree': 4,
                                'Neutral': 3,
                                'Disagree': 2,
                                'Strongly Disagree': 1})

replace_map = MappingProxyType({'Diasgree': 'Disagree',
               'Kako': 'Waia/Kako',
               'Mtito': 'Mtito Andei'})

reverse_map = MappingProxyType({v: k for k, v in ordinal_map.items()})


def engineer_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the change in values over time.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data to engineer features for.

    Returns
    -------
    df : pandas.DataFrame
        The data with the new features added.

    Notes
    -----
    - The features are the change in values after project implementation.
    - The features are calculated for each indicator with 'before' and 'after'.

    Example
    -------
    >>> df = engineer_change_features(df)

    """
    dataframe = df.copy()

    # Create a list of the columns to use
    for col in dataframe.columns:
        if col.endswith('_before') or col.endswith('_after'):
            base_col = col.rsplit('_', 1)[0]
            before_col = base_col + '_before'
            after_col = base_col + '_after'
            if before_col in dataframe.columns and after_col in dataframe.columns:
                change_col = base_col + '_change'
                dataframe[change_col] = dataframe[after_col] - dataframe[before_col]
                change_col_index = max(dataframe.columns.get_loc(before_col), dataframe.columns.get_loc(after_col)) + 1
                change_col_data = dataframe[change_col]
                dataframe.drop(columns=[change_col], inplace=True)
                dataframe.insert(change_col_index, change_col, change_col_data)
    return dataframe


def create_composite_feature(df, features, new_feature_name,
                             method='pca', weights=None, drop_features=False,
                             ordinal_map=ordinal_map):
    """
    Create a composite feature from a list of features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe containing the features.
    features : list
        The list of features to combine.
    new_feature_name : str
        The name of the new feature.
    drop_features : bool, default False
        Whether to drop the features used to create the new feature.
    method : str, default 'mean'
        The method to use to combine the features. Options are 'mean', 'sum',
        'weighted_mean', 'weighted_sum', 'interaction', and 'pca'.
    weights : list, default None
        The weights to use for the weighted_mean and weighted_sum methods.
    ordinal_map : dict, default None
        The mapping to use for ordinal features.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the new feature added.
    """
    data = df.copy()

    if ordinal_map is not None:
        dataframe = data.replace(ordinal_map)
    else:
        dataframe = data

    # if new_feature_name in dataframe.columns:
    #     raise ValueError(f'Feature {new_feature_name} already exists.')
    #
    # if not all(feature in dataframe.columns for feature in features):
    #     raise ValueError('Not all features in features exist in dataframe.')

    # if method == 'mean':
    #     dataframe[new_feature_name] = dataframe[features].mean(axis=1)
    #
    # elif method == 'sum':
    #     dataframe[new_feature_name] = dataframe[features].sum(axis=1)
    #
    # elif method == 'weighted_mean':
    #     if weights is None:
    #         raise ValueError('Weights must be provided for weighted_mean method.')
    #     dataframe[new_feature_name] = np.average(dataframe[features], axis=1, weights=weights)
    #
    # elif method == 'weighted_sum':
    #     if weights is None:
    #         raise ValueError('Weights must be provided for weighted_sum method.')
    #     dataframe[new_feature_name] = np.dot(dataframe[features], weights)
    #
    # elif method == 'interaction':
    #     dataframe[new_feature_name] = dataframe[features].prod(axis=1)
    #
    if method == 'pca':
        dataframe[features] = StandardScaler().fit_transform(dataframe[features])

        pca = PCA(n_components=1)

        dataframe[new_feature_name] = pca.fit_transform(dataframe[features])
    #
    # if drop_features:
    #     dataframe.drop(columns=features, inplace=True)

    return dataframe


def transform_features(df, features,
                       treat_outliers='drop', treat_skewness=None):
    """
    Transform features in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe containing the features.

    features : list
        The list of features to transform.

    treat_outliers : any, default None
        How to treat outliers. If None, no treatment is applied.

    treat_skewness : any, default None
        How to treat skewness. If None, no treatment is applied.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the transformed features.
    """
    dataframe = df.copy()

    if not all(feature in dataframe.columns for feature in features):
        raise ValueError('Not all features in features exist in dataframe.')

    if features is None:
        features = dataframe.columns.tolist()

    if not isinstance(features, list):
        features = list(features)

    for feature in features:

        if treat_outliers is not None:
            outliers = dataframe[(np.abs(stats.zscore(dataframe[feature])) > 3)]

            if treat_outliers == 'drop':
                dataframe.drop(outliers.index, inplace=True)

            elif treat_outliers == 'mean':
                dataframe[feature] = np.where(np.abs(stats.zscore(dataframe[feature])) > 3,
                                              dataframe[feature].mean(),
                                              dataframe[feature])

            elif treat_outliers == 'median':
                dataframe[feature] = np.where(np.abs(stats.zscore(dataframe[feature])) > 3,
                                              dataframe[feature].median(),
                                              dataframe[feature])

            elif treat_outliers == 'winsorize':
                dataframe[feature] = stats.mstats.winsorize(dataframe[feature], limits=0.03)

        if treat_skewness is not None:
            const = np.abs(dataframe[feature].min()) + 1

            if treat_skewness == 'cuberoot':
                dataframe[f'{feature} (cbrt)'] = np.cbrt(dataframe[feature])

            elif treat_skewness == 'sqrt':
                dataframe[f'{feature} (sqrt)'] = np.sqrt(dataframe[feature] + const)

            elif treat_skewness == 'log':
                dataframe[f'{feature} (log)'] = np.log(dataframe[feature] + const)

            elif treat_skewness == 'boxcox':
                dataframe[f'{feature} (boxcox)'] = stats.boxcox(dataframe[feature] + const)[0]

    return dataframe


def create_technology_features(df):
    """
    This function creates technology features from a given dataframe. It specifically looks for columns that start with 'trained' or 'adopted' and creates new columns based on these. The new columns are then used to create a melted dataframe with 'Technology' and 'Status' as the variable and value columns respectively.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the new technology features added.
    """
    # Create a copy of the dataframe to avoid modifying the original one
    dataframe = df.copy()

    # Identify the technology columns (those that start with 'trained' or 'adopted')
    technology_cols = [col for col in dataframe.columns if col.startswith('trained') or col.startswith('adopted')]

    # Iterate over the dataframe columns
    for col in dataframe.columns:

        # If the column is a technology column
        if col in technology_cols:
            # Split the column name to get the technology name
            technology_col = col.split('_', 1)[1]
            # Create the column names for 'trained' and 'adopted'
            trained_col = 'trained_' + technology_col
            adopted_col = 'adopted_' + technology_col

            # If the column starts with 'trained', replace 'Yes' with 'Trained' and 'No' with 'Not Trained'
            if col.startswith('trained'):
                dataframe.loc[:, trained_col] = dataframe[trained_col].replace({'Yes': 'Trained', 'No': 'Not Trained'})
            # If the column starts with 'adopted', replace 'Yes' with 'Adopted' and 'No' with 'Not Adopted'
            elif col.startswith('adopted'):
                dataframe.loc[:, adopted_col] = dataframe[adopted_col].replace({'Yes': 'Adopted', 'No': 'Not Adopted'})

    # Melt the dataframe to create a long format dataframe with 'Technology' and 'Status' columns
    dataframe = dataframe.melt(id_vars=[col for col in dataframe.columns if col not in technology_cols],
                               value_vars=technology_cols, var_name='Technology', value_name='Status')

    # Clean up the 'Technology' column by splitting on underscore and replacing it with a space
    dataframe.loc[:, 'Technology'] = dataframe['Technology'].apply(lambda x: x.split('_', 1)[1].replace('_', ' ').title())

    return dataframe


def calculate_percentages(df):
    """
    This function calculates the relative proportions of each category in a dataframe and returns the dataframe with the proportions expressed as percentages.

    The function assumes that the dataframe contains categorical data with the following categories: 'Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the relative proportions of each category expressed as percentages.
    """
    # Divide each value in the dataframe by the sum of the row it belongs to and multiply by 100 to get percentages
    df = df.div(df.sum(axis=1), axis=0) * 100

    # Define the order of the columns
    columns = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']

    # Reindex the dataframe to match the order of the columns and fill missing values with 0
    df = df.reindex(columns=columns, fill_value=0)

    # Round the values in the dataframe to 2 decimal places
    df = df.round(2)

    return df


def bar_plot(df, ax, palette=None):
    """
    This function plots a stacked bar plot of the relative proportions of each category in a dataframe.

    The function assumes that the dataframe contains categorical data with the following categories: 'Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'. Each category is represented by a different color in the bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe. It should contain the relative proportions of each category expressed as percentages.
    ax : matplotlib.axes.Axes
        The axes object to draw the plot onto.
    palette : list, optional
        The color palette to use for the different categories. If None, the 'Spectral' color palette from seaborn is used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the bar plot drawn onto it.
    """
    if palette is None:
        palette = sns.color_palette('Spectral', 5)
        palette.reverse()
    # Add annotations
    ax.barh(df.index, df['Strongly Disagree'], color=palette[4], label='Strongly Disagree')
    # Annotate Strongly Disagree
    for i, v in enumerate(df['Strongly Disagree']):
        if v > 0:
            ax.text(v + 0.5, i - 0.1, f"{v:.1f}%", color='black', fontsize=6)
    ax.barh(df.index, df['Disagree'], left=df['Strongly Disagree'], color=palette[3], label='Disagree')
    for i, v in enumerate(df['Disagree']):
        if v > 0:
            ax.text(v + df['Strongly Disagree'][i] + 0.5, i - 0.1, f"{v:.1f}%", color='black', fontsize=6)
    ax.barh(df.index, df['Neutral'], left=df['Strongly Disagree'] + df['Disagree'], color=palette[2], label='Neutral')
    for i, v in enumerate(df['Neutral']):
        if v > 0:
            ax.text(v + df['Strongly Disagree'][i] + df['Disagree'][i] + 0.5, i - 0.1, f"{v:.1f}%", color='black', fontsize=6)
    ax.barh(df.index, df['Agree'], left=df['Strongly Disagree'] + df['Disagree'] + df['Neutral'], color=palette[1], label='Agree')
    for i, v in enumerate(df['Agree']):
        if v > 0:
            ax.text(v + df['Strongly Disagree'][i] + df['Disagree'][i] + df['Neutral'][i] + 0.5, i - 0.1, f"{v:.1f}%", color='black', fontsize=6)
    ax.barh(df.index, df['Strongly Agree'], left=df['Strongly Disagree'] + df['Disagree'] + df['Neutral'] + df['Agree'], color=palette[0], label='Strongly Agree')
    for i, v in enumerate(df['Strongly Agree']):
        if v > 0:
            ax.text(v + df['Strongly Disagree'][i] + df['Disagree'][i] + df['Neutral'][i] + df['Agree'][i] + 0.5, i - 0.1, f"{v:.1f}%", color='black', fontsize=6)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xticklabels(np.arange(0, 101, 25))
    ax.set_xlabel('Relative Proportion (%)')
    ax.set_ylabel('')
    ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.275),
              facecolor='gray', framealpha=0.05)

    return ax


def conduct_hypothesis_test(df, column):
    """
    This function conducts a hypothesis test on a given dataframe and column. It performs a t-test if there are two unique values in the column, and an ANOVA if there are more than two unique values.

    The function assumes that the dataframe contains categorical data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    column : str
        The column to conduct the hypothesis test on.

    Returns
    -------
    None
    """
    for col in df.select_dtypes('category').columns:
        unique_values = df[col].unique()

        # Perform t-test if there are two unique values
        if len(unique_values) == 2:
            group1 = df[df[col] == unique_values[0]][column]
            group2 = df[df[col] == unique_values[1]][column]
            t_stat, p_value = ttest_ind(group1, group2)
            print(f"T-Test for {col}")
            print(f"-----------{'-' * len(col)}")
            print("H0: The means of the two groups are equal.")
            print("H1: The means of the two groups are not equal.\n")

            display(ttest_ind(group1, group2))

        # # Display the table of results
            display(pd.DataFrame({'Group 1': [group1.mean(), group1.std(), group1.count()],
                                    'Group 2': [group2.mean(), group2.std(), group2.count()],
                                    'Difference': [group1.mean() - group2.mean(), np.nan, np.nan]},
                                     index=['Mean', 'Standard Deviation', 'Sample Size']))

            print(f"t-statistic (333) = {t_stat:.4f}")
            print(f"p-value = {p_value:.4f}\n")
            if p_value < 0.05:
                print(f"Reject H0. The means of the two groups are not equal.\n")
            else:
                print(f"Fail to reject H0. The means of the two groups are equal.\n")

        # Perform ANOVA if there are more than two unique values
        elif len(unique_values) > 2:
            model = pg.anova(data=df, dv=column, between=col)
            print(f"ANOVA for {col}")
            print(f"----------{'-' * len(col)}")
            print("H0: The means of the groups are equal.")
            print("H1: The means of the groups are not equal.\n")
            display(model)
            print(f"F-statistic ({model['ddof1'][0]}, {model['ddof2'][0]}) = {model['F'][0]:.4f}")
            print(f"p-value = {model['p-unc'][0]:.4f}\n")
            if model['p-unc'][0] < 0.05:
                print(f"Reject H0: The means of the groups are not equal.\n")
                print(f"Post-hoc test for {col}")
                print(f"------------------{'-' * len(col)}")
                posthoc = pg.pairwise_tukey(data=df, dv=column, between=col)
                display(posthoc)
            else:
                print(f"Fail to reject H0: The means of the groups are equal.\n")


def filter_outliers(dataframe: pd.DataFrame, columns: list):
    """
    This function filters outliers from the specified columns of a dataframe.

    Outliers are defined as values that are more than three standard deviations away from the mean. The function calculates the mean and standard deviation for each specified column, and then removes rows where the column value is an outlier.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe from which outliers are to be removed.
    columns : list
        The list of column names to check for outliers.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with outliers removed.
    """
    for col in columns:
        # Calculate the mean and standard deviation of the column
        mean = dataframe[col].mean()
        std = dataframe[col].std()

        # Define the limit for outlier detection as three standard deviations from the mean
        limit = std * 3

        # Calculate the lower and upper bounds for outlier detection
        lower, upper = (mean - limit), (mean + limit)

        # Filter the dataframe to remove outliers
        dataframe = dataframe[(dataframe[col] > lower) & (dataframe[col] < upper)]

    return dataframe


def transform_skewed_features(dataframe: pd.DataFrame, columns: list):
    """
    This function transforms skewed features in a dataframe.

    The function assumes that the dataframe contains numerical features with skewed distributions. It transforms the features using the log transformation.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the skewed features.
    columns : list
        The list of column names to transform.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the skewed features transformed.
    """
    for col in columns:
        # Transform the feature using the log transformation
        dataframe[col] = np.log(dataframe[col])

    return dataframe


def plot_distribution(dataframe: pd.DataFrame,
                      feature: str,
                      by: str = None,
                      histyle: str = 'dodge',
                      violin: bool = False,
                      fh: int = 4,
                      hr: int = 1,
                      log: bool = False,
                      **kwargs):
    """
    This function plots the distribution of a specified numerical feature in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the feature to plot.
    feature : str
        The name of the feature to plot.
    by : str, optional
        The name of the column to group the data by. The default is None.
    histyle : str, optional
        The style of the histogram. The default is 'dodge'.
    violin : bool, optional
        Whether to plot a violin plot. The default is False.
    kde : bool, optional
        Whether to plot a kernel density estimate plot. The default is True.
    fh : int, optional
        The height of the figure. The default is 4.
    hr : int, optional
        The height ratio of the subplots. The default is 1.
    log : bool, optional
        Whether to plot the y-axis on a log scale. The default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the seaborn plotting functions.

    Examples
    --------
    >>> plot_distribution()
    """
    fig, ax = plt.subplots(2, 1, figsize=(6, fh),
                           gridspec_kw={'height_ratios': [7.5, hr]},
                           sharex=True)
    sns.histplot(data=dataframe, x=feature, hue=by, multiple=histyle, ax=ax[0], shrink=0.8, log_scale=log, **kwargs)

    if violin:
        sns.violinplot(data=dataframe, x=feature, y=by, ax=ax[1], **kwargs)
    else:
        sns.boxplot(data=dataframe, x=feature, y=by, ax=ax[1], width=0.5, **kwargs)
    ax[1].set_ylabel('')
    ax[1].set_xlabel(feature.replace('_', ' ').title())

    return fig, ax
