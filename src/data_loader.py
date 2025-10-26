
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(os.getenv('ROOT', '.')).expanduser()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

# demo courses
excluded_demo_courses = [24, 28, 95, 98, 158, 165, 169, 184, 192, 193, 198, 201]

# introductory course (not adaptive)
excluded_introductory_courses = [50, 52, 90, 129, 153, 181]

# student users
students = [30,   31,   32,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   48,   50,   51,   52,   53,   54,
         55,   56,   57,   59,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,
         81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   98,   99,  100,  101,  102,  103,
        104,  105,  106,  107,  108,  109,  110,  111,  113,  114,  115,  116,  117,  118,  119,  120,  122,  124,  125,  126,  127,  128,
        129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140,  141,  142,  143,  144,  145,  146,  147,  148,  149,  150,
        151,  152,  153,  154,  155,  156,  157,  158,  159,  160,  162,  163,  164,  165,  167,  174,  180,  189,  190,  191,  192,  193,
        196,  198,  200,  204,  205,  206,  207,  208,  209,  210,  214,  216,  220,  222,  223,  224,  225,  226,  227,  228,  229,  231,
        232,  234,  236,  237,  240,  242,  247,  251,  252,  255,  258,  259,  260,  261,  264,  265,  267,  269,  270,  274,  277,  278,
        283,  285,  287,  290,  291,  292,  293,  294,  295,  297,  298,  300,  301,  302,  303,  304,  306,  311,  312,  313,  315,  316,
        318,  319,  322,  323,  324,  325,  326,  331,  333,  334,  336,  337,  338,  339,  340,  341,  342,  343,  345,  347,  348,  349,
        350,  351,  352,  353,  354,  356,  357,  358,  359,  361,  366,  367,  368,  399,  518,  523,  581,  585,  586,  590,  591,  593,
        595,  605,  607,  627,  646,  647,  648,  649,  650,  651,  652,  653,  654,  655,  656,  657,  658,  659,  660,  661,  662,  665,
        666,  667,  668,  672,  673,  674,  675,  676,  677,  678,  679,  691,  692,  693,  695,  696,  697,  705,  728,  740,  754,  767,
        768,  769,  770,  771,  772,  773,  775,  776,  777,  778,  779,  780,  782,  783,  784,  785,  786,  787,  788,  789,  790,  791,
        793,  794,  796,  797,  798,  800,  801,  805,  806,  807,  808,  809,  811,  812,  813,  815,  816,  817,  818,  820,  821,  822,
        823,  824,  825,  826,  827,  828,  829,  830,  831,  832,  833,  834,  835,  836,  837,  838,  839,  841,  842,  843,  844,  845,
        846,  848,  849,  850,  851,  853,  854,  855,  857,  858,  859,  860,  861,  862,  863,  864,  865,  866,  867,  868,  869,  870,
        871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,  882,  884,  885,  886,  887,  888,  889,  890,  891,  892,  893,
        894,  896,  899,  901,  902,  904,  905,  906,  907,  910,  912,  915,  916,  918,  921,  923,  924,  927,  928,  929,  931,  932,
        935,  936,  937,  938,  939,  940,  949,  951,  952,  954,  956,  959,  960,  964,  970,  971,  974,  978,  979,  980,  983,  984,
        985,  986,  987,  988,  989,  990,  991,  992,  993,  994,  995,  996,  998,  999, 1002, 1003, 1005, 1006, 1007, 1009, 1010, 1011,
       1012, 1013, 1018, 1019, 1021, 1022, 1023, 1024, 1025, 1028, 1029, 1030, 1035, 1039, 1043, 1044, 1046, 1048, 1049, 1050, 1055, 1056,
       1057, 1058, 1059, 1060, 1061, 1063, 1064, 1065, 1069, 1070, 1071, 1072, 1073, 1074, 1077, 1081, 1087, 1092, 1094, 1095, 1096, 1097,
       1098, 1099, 1101, 1104, 1106, 1108, 1109, 1111, 1113, 1114, 1119, 1121, 1131, 1132, 1133, 1134, 2757, 3254]

def load_quiz(path: str) -> pd.DataFrame:
    """Reads quiz data.

    Args:
        path: Quiz filepath (.xlsx format).

    Returns:
        A Pandas dataframe.
    """
    df_quiz = pd.read_excel(path)
    df_quiz = df_quiz.assign(quiz_dropout=df_quiz['TestAbschluss'].apply(lambda x: x == 'inprogress'))
    df_quiz.rename(columns= {
        "KursID": "courseid", 
        "StudID": "userid", 
        "VersuchsID": "attemptid", 
        "VersuchsNr": "attemptnr", 
        "PunkteErreicht": "points",
        "MaxPunkte": "maxpoints"
    }, inplace=True)
    
    df_quiz = df_quiz.loc[
        (~df_quiz.courseid.isin(excluded_demo_courses)) &
        (~df_quiz.courseid.isin(excluded_introductory_courses)) &
        (df_quiz.userid.isin(students))
    ]
    
    df_quiz = df_quiz.assign(is_preknowledge_test=df_quiz.TestName.str.contains('Standort'))
    df_quiz = df_quiz.groupby(['courseid','userid'], as_index=False).agg(**{
        'test_names': ('TestName', list),
        'contains_preknowledge_test': ('is_preknowledge_test', any)
    }).merge(df_quiz, on=['courseid','userid'])

    bad_data =[] # users with no preknowledge tests
    bad_data2 = [] # users with preknowledge test with score NaN
    def func(x):
        """
            this function modifies and populates bad_data and bad_data2 from global scope
        """
        # get indices of preknowledge_tests, some (userid, courseid)s have more than one preknowledge tests
        indices = [idx for idx in x.index if x[idx]] 
        if indices:
            points = df_quiz.loc[indices, 'points']
            maxpoints = df_quiz.loc[indices, 'maxpoints']
            try:
                r = (points/maxpoints).mean()
            except ZeroDivisionError:
                # print(f"ZeroDivisionError in points/maxpoints for indices: {indices}, points:{points}  maxpoints: {maxpoints}")
                r = np.nan
            finally:
                if r is np.nan:
                    bad_data2.append(df_quiz.loc[indices])
                    # print(f'something wrong with indices: {indices}, points {points.tolist()}, maxpoints: {maxpoints.tolist()}')
        else:
            # no preknowledge test available
            bad_data.append(df_quiz.loc[x.index])
            # print(f'no preknowledge test available for indices: {x.index.tolist()}')
            r = np.nan
        return r
    
    df_quiz = df_quiz.groupby(['courseid','userid'], as_index=False).agg(**{
        'preknowledge_test_score':('is_preknowledge_test', func),
    }).merge(df_quiz, on=['courseid', 'userid'])

    # remove bad data
    bad_data_df = pd.concat(bad_data, axis=0)
    excluded_indices = bad_data_df.index
    df_quiz = df_quiz.loc[~df_quiz.index.isin(excluded_indices)]
    bad_data2_df = pd.concat(bad_data2, axis=0)
    excluded_indices = bad_data2_df.index
    df_quiz = df_quiz.loc[~df_quiz.index.isin(excluded_indices)]
    
    # remove preknowledge tests
    df_quiz = df_quiz.loc[~df_quiz.is_preknowledge_test]

    def f(x):
        try:
            semester = re.search(r'(HS[0-9]{2}\/[0-9]{2}|FS[0-9]{2})', x).group()
        except Exception as e:
            # print(e, x)
            semester = None
        return semester
    
    # extracting semester 
    df_quiz = df_quiz.assign(semester=lambda df_: df_.CourseShortname.apply(f))
    df_quiz = df_quiz.drop(columns=['test_names', 'CourseFullname', 'CourseShortname', 'TestName'])
    return df_quiz

def load_log(paths: list[str]) -> pd.DataFrame:
    """Reads log data.

    Args:
        paths: log filepaths.

    Returns:
        A Pandas dataframe.
    """
    df_log = pd.concat(
        [pd.read_csv(path, delimiter=';') for path in paths], 
        axis=0, 
        ignore_index=True
    )
    df_log = df_log.assign(datetime=df_log['timecreated'].apply(pd.to_datetime, unit='s'))
    
    # remove demo courses
    df_log = df_log.loc[~(df_log.courseid.isin(excluded_demo_courses))]
    # remove introductory course (not adaptive)
    df_log = df_log.loc[~(df_log.courseid.isin(excluded_introductory_courses))]
    # keep only student users
    df_log = df_log.loc[df_log.userid.isin(students)]
    df_log = df_log.drop(columns=["ip","realuserid"])
    return df_log

def merge_tables(df_quiz: pd.DataFrame, df_log: pd.DataFrame) -> pd.DataFrame:
    """Merges quiz and log data.

    Args:
        df_quiz: quiz dataframe.
        df_log: log dataframe.

    Returns:
        A Pandas dataframe.
    """

    # merging two tables
    df_merged = df_log.merge(
        df_quiz,
        left_on=['courseid','objectid','userid'],
        right_on=['courseid', 'attemptid','userid']
    )
    
    # estimating students' start of quizzes
    a = df_merged.loc[df_merged.eventname=='\\mod_quiz\\event\\attempt_started', 
        ['courseid', 'userid', 'TestID', 'attemptid', 'datetime']]
    a = a.rename(columns={'datetime':'student_start_of_quiz'})
    if 'student_start_of_quiz' in df_merged.columns:
        del df_merged['student_start_of_quiz']
    df_merged = df_merged.merge(a, on=['courseid', 'userid', 'TestID', 'attemptid'])
    
    # estimating start of quizzes
    a = df_merged.groupby(['courseid', 'TestID'], as_index=False).agg(start_of_quiz=('student_start_of_quiz', lambda x: x.min()))
    if 'start_of_quiz' in df_merged.columns:
        del df_merged['start_of_quiz']
    df_merged = df_merged.merge(a, on=['courseid', 'TestID'])
    
    # estimating start of courses
    a = df_merged.groupby('courseid', as_index=False).agg(start_of_course=('start_of_quiz', lambda x: x.min()))
    a = a.assign(end_of_course=lambda x: x.start_of_course + pd.DateOffset(months=8))
    if 'start_of_course' in df_merged.columns:
        del df_merged['start_of_course']
    df_merged = df_merged.merge(a, on='courseid')
    
    # only keeping activities during the semester [start_of_cours, start_of_course + 6 months]
    df_merged = df_merged.loc[df_merged['datetime'] <= df_merged['end_of_course']]

    EVENT_SUBMIT = '\\mod_quiz\\event\\attempt_submitted'
    # EVENT_START = '\\mod_quiz\\event\\attempt_started'
    
    def func(referenced_dataframe=None, event=None):
        def f(s):
            if (s == event).any():
                idx = s[s == event].idxmax()
                return referenced_dataframe.loc[idx, 'datetime']
            return pd.to_datetime('2050')
        return f
    
    a = df_merged.groupby(['courseid', 'userid', 'objectid'], as_index=False).agg(
        submission_time=('eventname', func(referenced_dataframe=df_merged, event=EVENT_SUBMIT))
    )
    if 'submission_time' in df_merged.columns:
        del df_merged['submission_time']
    df_merged = df_merged.merge(a, on=['courseid', 'userid', 'objectid'])
    df_merged.loc[:, 'event_before_submission'] = df_merged['datetime'] <= df_merged['submission_time']

    def func(x):
        evs = [r for r in x]
        assert all([type(r) == str for r in evs]), f'some events are not str: {evs}'
        return any(['attempt_submitted' in r for r in evs])
    
    df_merged = df_merged.groupby(['courseid', 'userid', 'objectid'], as_index=False).agg(**{
        'any_submission_event': ('eventname', func)
    }).merge(df_merged, on=['courseid', 'userid', 'objectid'])

    return df_merged

if __name__ == '__main__':

    # loading quiz data
    path = os.path.join(ROOT, 'data/raw/ALMoo Quiz attempts.xlsx')
    df_quiz = load_quiz(path)
    print(df_quiz)

    # loading log data
    paths = [
        os.path.join(ROOT, 'data/raw/mdl_logstore_standard_log (mod_quiz Group 1).csv'),
        os.path.join(ROOT, 'data/raw/mdl_logstore_standard_log (mod_quiz Group 2).csv'),
        os.path.join(ROOT, 'data/raw/mdl_logstore_standard_log (mod_quiz Group 3).csv')
    ]
    df_log = load_log(paths)
    print(df_log)

    df_merged = merge_tables(df_quiz, df_log)
    print(df_merged)

    df_merged.to_csv(os.path.join(ROOT, "data/preprocessed/clean_merged_data.csv"), index=True, index_label="original_index")
    
