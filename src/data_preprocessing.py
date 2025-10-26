import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
import os
from sklearn.preprocessing import StandardScaler
load_dotenv()
ROOT = Path(os.getenv('ROOT', '.')).expanduser()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
    
target = 'did_submit'
feature_names_all = [
    # counts of activities
    'all_activities_count', 'attempt_viewed_count', 'course_module_viewed_count', 
    # # counts of time_of_day
    # 'morning_count', 'afternoon_count', 'evening_count', 
    # # counts of day_type
    # 'workday_count', 'weekend_count',
    # counts of day_type_time_of_day
    'workday_morning_count', 'workday_afternoon_count', 'workday_evening_count', 
    'weekend_morning_count', 'weekend_afternoon_count', 'weekend_evening_count',
    # counts of inactivity
    'days_inactive_since_last_activity', 
    'days_inactive', 
    # days since start
    'date_rel',
    # attemptnr
    'attemptnr',
    # previous attempts
    'previous_attempts',
    # previous performance
    'previous_perf',
    # stats
    'stat_min', 'stat_max', 'stat_mean', 'stat_median', 'stat_sd', 'stat_skew', 'stat_kurtosis'
]
excluded_features = [
    'all_activities_count', 
    'attempt_viewed_count', 
    'course_module_viewed_count', 
    'days_inactive_since_last_activity',
    # 'days_inactive',
    'date_rel',
    'stat_max',
    # 'kurtosis'
]

feature_names = [feat for feat in feature_names_all if feat not in excluded_features]

def labeling_3(df: pd.DataFrame) -> pd.DataFrame:
    """Create labels for the ML model.
    Args:
        df: The input DataFrame.

    Returns:
        A dataframe with label ``did_submit``.
    """
    return df.loc[(df.date_rel >= 0)&(df.date_rel<20)].assign(
        did_submit=lambda df_: df_.days_before_submit == 0
    )
    
def split_data(
        df: pd.DataFrame, 
        train_semesters: list[str], 
        test_semesters: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and testing subsets by semester.

    Args:
        df:
            The input DataFrame containing at least a ``semester`` column.
        train_semesters:
            List of semester identifiers to include in the training set.
        test_semesters:
            List of semester identifiers to include in the test set.

    Returns:
        A tuple ``(train_data, test_data)`` where:
            - ``train_data``: All rows whose ``semester`` is in ``train_semesters``.
            - ``test_data``: All rows whose ``semester`` is in ``test_semesters``.
    """
    train_data = df.loc[df.semester.isin(train_semesters)]
    test_data = df.loc[(df.semester.isin(test_semesters))]
    
    # train_data, test_data = train_test_split(df, test_size=0.2)

    # other preprocessing
    # train_data = train_data.groupby(['courseid','userid','TestID','attemptid'], as_index=False).last()
    # test_data = test_data.groupby(['courseid','userid','TestID','attemptid'], as_index=False).last()
    # test_data = test_data.groupby(['courseid','userid','TestID','attemptid']).apply(lambda group: group.iloc[-2:]).reset_index(drop=True)
    return train_data, test_data

def plot_feature_correlation(
    df: pd.DataFrame,
    target: str,
    feature_names: list[str],
    save_path: str | None = None,
    title: str = "Feature Correlation Heatmap",
    cmap: str = "coolwarm",
    figsize: tuple[int, int] = (8, 7),
) -> None:
    """Compute and display a correlation heatmap between target and features.

    Args:
        df: Input DataFrame.
        target: Target column name to include in correlation.
        feature_names: List of feature column names.
        save_path: Optional file path to save the correlation plot image.
        title: Title for the plot.
        cmap: Colormap to use for the heatmap.
        figsize: Size of the matplotlib figure (width, height).

    """
    # Subset DataFrame and compute correlation
    corr = df[[target] + feature_names].corr()

    # Mask upper triangle (to avoid duplicate values)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Plot
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            fmt=".2f",
            cmap=cmap,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Save if path is provided
        if save_path:
            save_path = os.path.expanduser(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"✅ Correlation plot saved to: {save_path}")

        plt.show()
        plt.close(fig)

def save_data(X, y, filename, groups=None):
    """Save feature, label, and optional group arrays as a single NumPy file.

    Args:
        X: Feature matrix as a NumPy array.
        y: Label array.
        filename: Name of the file to save (within the preprocessed data folder).
        groups: Optional array of group identifiers to include.
    """
    with open(filename, 'wb') as f:
        if groups is not None:
            np.save(f, np.concatenate((X, y[:, np.newaxis], groups[:,np.newaxis]), axis=1) )
        else:
            np.save(f, np.concatenate((X, y[:, np.newaxis]), axis=1) )

def load_data(filename, with_groups=False):
    """Load preprocessed NumPy data into feature, label, and optional group arrays.

    Args:
        filename: Name of the file to load (within the preprocessed data folder).
        with_groups: Whether to return group identifiers as the third output.

    Returns:
        Tuple of arrays: (X, y) or (X, y, groups) depending on `with_groups`.
    """
    with open(filename, 'rb') as f:
        data = np.load(f)
        if with_groups:
            return data[:,:-2], data[:,-2], data[:,-1]
        else:
            return data[:,:-1], data[:,-1]

def preprocess(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the merged data (https://arxiv.org/pdf/2507.02681?).

    Args:
        df_merged: merged data.

    Returns:
        A Pandas dataframe.
    """
    
    df_merged['eventname'] = df_merged['eventname'].str.split('\\').str[-1]
    # df_merged = df_merged.join(df_merged['eventname'].str.get_dummies())
    df_merged = (df_merged
        .assign(
            date_rel=lambda df_: pd.to_timedelta(df_.datetime.dt.date - df_.student_start_of_quiz.dt.date).dt.days,
            submission_date_rel = lambda df_: pd.to_timedelta(df_.submission_time.dt.date - df_.student_start_of_quiz.dt.date).dt.days
        )
    )
    df_merged = df_merged.loc[df_merged['datetime'] <= df_merged['submission_time'] ]
    
    def day_type_time_of_day(dt):
        if dt.day_of_week >= 5:
            if 5 <= dt.hour < 12:
                return 'weekend_morning'
            elif 12 <= dt.hour < 18:
                return 'weekend_afternoon'
            else:
                return 'weekend_evening'
        else:
            if 5 <= dt.hour < 12:
                return 'workday_morning'
            elif 12 <= dt.hour < 18:
                return 'workday_afternoon'
            else:
                return 'workday_evening'
        
    def weekend_or_workday(dt):
        if dt.day_of_week >= 5:
            return 'weekend'
        else:
            return 'workday'
    
    def morning_or_evening(dt):
        if 5 <= dt.hour < 12:
            return 'morning'
        elif 12 <= dt.hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    df_merged = df_merged.assign(
        # day_type=df_merged['datetime'].apply(weekend_or_workday),
        # time_of_day=df_merged['datetime'].apply(morning_or_evening),
        day_type_time_of_day=lambda df_: df_['datetime'].apply(day_type_time_of_day)
    )
    
    # Function to count previous attempts and previous submitted attempts within a group
    def count_previous_attempts_in_group(group):
        group = group.sort_values(by='student_start_of_quiz')
        group['previous_attempts'] = 0
        group['previous_submitted_attempts'] = 0
        for i in range(1, len(group)):
            previous_attempts = group.iloc[:i]
            previous_submitted_attempts = previous_attempts[previous_attempts['submission_time'] < group.iloc[i]['student_start_of_quiz']]
            group.at[group.index[i], 'previous_attempts'] = previous_attempts['attemptid'].nunique()
            group.at[group.index[i], 'previous_submitted_attempts'] = previous_submitted_attempts['attemptid'].nunique()
        return group
    
    # Group by courseid, userid, semester, and attemptid to get unique attempts
    unique_attempts = df_merged.groupby(['courseid', 'userid', 'TestID', 'semester', 'attemptid']).agg(
        student_start_of_quiz=('student_start_of_quiz', 'first'),
        submission_time=('submission_time', 'first')
    ).reset_index()
    
    # Group by courseid, userid, semester and apply the counting function on unique attempts
    res = unique_attempts.groupby(['courseid', 'userid', 'semester']).apply(count_previous_attempts_in_group).reset_index(drop=True)
    
    # Merge the counts back into the original dataframe
    df_merged = df_merged.merge(res[['courseid', 'userid', 'TestID', 'semester', 'attemptid','previous_attempts','previous_submitted_attempts']],
                on=['courseid', 'userid', 'TestID', 'semester', 'attemptid'])
    
    del res
    df_merged = df_merged.assign(
        is_first_attempt=lambda df_: df_['previous_attempts'] == 0,
        previous_perf=lambda df_: np.where(df_['is_first_attempt'], 0.5, df_['previous_submitted_attempts']/df_['previous_attempts'])
    )

    previous_perf_median = df_merged.loc[~df_merged.is_first_attempt, 'previous_perf'].median()
    df_merged.loc[lambda df_: df_.is_first_attempt, 'previous_perf'] = previous_perf_median

    
    # events_included = ['attempt_viewed', 'attempt_summary_viewed','course_module_viewed']
    events_included = ['attempt_started', 'attempt_viewed', 'attempt_summary_viewed', 'course_module_viewed']
    
    df = df_merged.loc[
        df_merged.eventname.isin(events_included)
    ].groupby(["courseid","userid","TestID","attemptid","attemptnr","semester","date_rel","submission_date_rel", "student_start_of_quiz",
              'previous_attempts','previous_submitted_attempts','is_first_attempt','previous_perf'],
              as_index=False).agg(
        activity_in_day_datetime=("datetime",list),
        activity_in_day_eventname=("eventname",list),
        # activity_in_day_day_type=('day_type', list),
        # activity_in_day_time_of_day=('time_of_day', list)
        activity_in_day_day_type_time_of_day=('day_type_time_of_day',list)
    )
    
    df = df.assign(
        days_before_submit=lambda df_: df_.submission_date_rel - df_.date_rel
    )
    
    # Function to fill in missing days
    def fill_missing_days(group, max_days=60):
        min_date_rel = group['date_rel'].min()
        max_date_rel = group['date_rel'].max()
        if max_date_rel - min_date_rel > max_days:
            min_date_rel = group.loc[(group['date_rel'] >= max_date_rel - max_days), 'date_rel'].iloc[0]
    
        group['is_added'] = False
        # Create records for missing days
        empty_records = []
        for day in range(min_date_rel, max_date_rel + 1):
            if day not in group['date_rel'].values:
                empty_record = {
                    'courseid': group['courseid'].iloc[0],
                    'userid': group['userid'].iloc[0],
                    'TestID': group['TestID'].iloc[0],
                    'attemptid': group['attemptid'].iloc[0],
                    'attemptnr': group['attemptnr'].iloc[0],
                    'semester': group['semester'].iloc[0],
                    'student_start_of_quiz': group['student_start_of_quiz'].iloc[0],
                    'previous_attempts': group['previous_attempts'].iloc[0],
                    'previous_submitted_attempts': group['previous_submitted_attempts'].iloc[0],
                    'is_first_attempt': group['is_first_attempt'].iloc[0],
                    'previous_perf': group['previous_perf'].iloc[0],
                    'date_rel': day,
                    'submission_date_rel': group['submission_date_rel'].iloc[0],
                    'days_before_submit': group['submission_date_rel'].iloc[0] - day,
                    'is_added': True,
                    'activity_in_day_datetime': [],
                    'activity_in_day_eventname': [],
                    # 'activity_in_day_day_type': [],
                    # 'activity_in_day_time_of_day': []
                    'activity_in_day_day_type_time_of_day': []
                }
                empty_records.append(empty_record)
    
        group = pd.concat((group, pd.DataFrame(empty_records)), ignore_index=True)
        
        return group.sort_values(by='date_rel').reset_index(drop=True)
    
    # Group by the relevant columns and apply the function
    df_filled = df.groupby(['courseid', 'userid', 'TestID', 'attemptid']).apply(fill_missing_days).reset_index(drop=True)
    
    # Function to calculate inactive days
    def calculate_inactive_days(group):
        group['days_inactive_since_last_activity'] = 0
        group['days_inactive'] = 0
        days_inactive_since_last_activity = 0
        days_inactive = 0
        
        for i, row in group.sort_values(by='date_rel').iterrows():
            
            group.at[i, 'days_inactive_since_last_activity'] = days_inactive_since_last_activity
            group.at[i, 'days_inactive'] = days_inactive
    
            if not row['activity_in_day_datetime']:
                days_inactive_since_last_activity += 1
                days_inactive += 1
            else:
                days_inactive_since_last_activity = 0
    
        return group
    
    df_filled = df_filled.groupby(['courseid', 'userid', 'TestID', 'attemptid']).apply(calculate_inactive_days).reset_index(drop=True)
    df_filled.head()

    
    # Function to accumulate data until each date_rel
    def accumulate_until_date_rel(group):
        group = group.sort_values(by='date_rel')
        accumulated_activity_datetime = []
        accumulated_activity_eventname = []
        # accumulated_activity_day_type = []
        # accumulated_activity_time_of_day = []
        accumulated_activity_day_type_time_of_day = []
        
        # Initialize new columns with empty lists
        group['accumulated_activity_datetime'] = [[] for _ in range(len(group))]
        group['accumulated_activity_eventname'] = [[] for _ in range(len(group))]
        # group['accumulated_activity_day_type'] = [[] for _ in range(len(group))]
        # group['accumulated_activity_time_of_day'] = [[] for _ in range(len(group))]
        group['accumulated_activity_day_type_time_of_day'] = [[] for _ in range(len(group))]
    
        for i, row in group.iterrows():
            accumulated_activity_datetime.extend(row['activity_in_day_datetime'])
            accumulated_activity_eventname.extend(row['activity_in_day_eventname'])
            # accumulated_activity_day_type.extend(row['activity_in_day_day_type'])
            # accumulated_activity_time_of_day.extend(row['activity_in_day_time_of_day'])
            accumulated_activity_day_type_time_of_day.extend(row['activity_in_day_day_type_time_of_day'])
            
            group.at[i, 'accumulated_activity_datetime'] = accumulated_activity_datetime.copy()
            group.at[i, 'accumulated_activity_eventname'] = accumulated_activity_eventname.copy()
            # group.at[i, 'accumulated_activity_day_type'] = accumulated_activity_day_type.copy()
            # group.at[i, 'accumulated_activity_time_of_day'] = accumulated_activity_time_of_day.copy()
            group.at[i, 'accumulated_activity_day_type_time_of_day'] = accumulated_activity_day_type_time_of_day.copy()
    
        return group
    
    
    # Apply the accumulation function
    df_acc = df_filled.groupby(['courseid', 'userid', 'TestID', 'attemptid']).apply(accumulate_until_date_rel).reset_index(drop=True)

    def func(x):
        if len(x) < 2:
            out = [timedelta(0)]
        else:
            out = [(t2-t1) for t1,t2 in zip(x[:-1], x[1:])]
        return out
        
    def compute_stats(df):
        df = df.assign(accumulated_activity_datetime_diff=lambda df_: 
            df_.accumulated_activity_datetime.apply(func)
        ).assign(accumulated_activity_datetime_diff_seconds=lambda df_:
            df_.accumulated_activity_datetime_diff.apply(lambda x: [r.seconds for r in x])
        )
        
        # for time_of_day in ['morning', 'afternoon', 'evening']:
        #     df[f"{time_of_day}_count"]=df.accumulated_activity_time_of_day.apply(lambda l: l.count(time_of_day))
        
        # for day_type in ['workday', 'weekend']:
        #     df[f"{day_type}_count"]=df.accumulated_activity_day_type.apply(lambda l: l.count(day_type))
    
        for day_type_time_of_day in ['workday_morning','workday_afternoon','workday_evening','weekend_morning','weekend_afternoon','weekend_evening']:
            df[f"{day_type_time_of_day}_count"] = df.accumulated_activity_day_type_time_of_day.apply(lambda l: l.count(day_type_time_of_day))
            
        df["all_activities_count"] = 0
        for ev in ['attempt_viewed', 'attempt_summary_viewed', 'course_module_viewed']:
            df[f"{ev}_count"]=df.accumulated_activity_eventname.apply(lambda l: l.count(ev))
            df["all_activities_count"] += df[f"{ev}_count"]
        
        STATS_EXTENDED = {
            'stat_min': np.min,
            'stat_max': np.max,
            'stat_mean': np.mean,
            'stat_median':np.median,
            'stat_sd': np.std,
            'stat_skew': skew,
            'stat_kurtosis': kurtosis,
        }
        
        print('calculating stats...')
        for stat, func_ in STATS_EXTENDED.items():
            df[stat] = df['accumulated_activity_datetime_diff_seconds'].apply(func_)
            
        df.fillna(0, inplace=True)
        return df
    
    df_stat = compute_stats(df_acc)

    return df_stat
    
if __name__ == '__main__':
    date_cols = [
        'datetime',
        'submission_time',
        'StartZeit',
        'EndZeit',
        'student_start_of_quiz',
        'start_of_quiz',
        'start_of_course',
        'end_of_course'
    ]

    df_merged = pd.read_csv(os.path.join(ROOT, "data/raw/clean_merged_data.csv"),
                           parse_dates=date_cols)
    df_stats = preprocess(df_merged)
    print(df_stats)
    
    train_semesters=['HS17/18', 'FS18', 'HS18/19']
    test_semesters=['FS19']
    train_data_, test_data_ = split_data(df_stats, train_semesters, test_semesters)
    print(train_data_, test_data_)

    train_data = labeling_3(train_data_)
    test_data = labeling_3(test_data_)

    # plot_feature_correlation(train_data, target, feature_names)
    
    X_train = train_data[feature_names].values
    y_train = train_data[target].values
    groups = train_data['attemptid'].values
    
    X_test = test_data[feature_names].values
    y_test = test_data[target].values

    # # Scale features using StandardScaler
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # saving preprocessed data
    save_data(X_train, y_train, os.path.join(ROOT, 'data/preprocessed/train-data.npy'), groups=groups)
    save_data(X_test, y_test, os.path.join(ROOT, 'data/preprocessed/test-data.npy'))
