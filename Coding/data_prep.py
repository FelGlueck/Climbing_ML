# Basic imports
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from collections import Counter


def mid(s, offset, amount):
    return s[offset:offset+amount]

def consGrade(cleaned_ascents):
    # This coding is used to calculate different grading schemes into one consecutive number
    # This could have a big effect especially for data from climbers that climb in many different locations

    prim_cleaned_ascents_df = cleaned_ascents.copy()

    # we are adding the primary and secondary grading scheme for each route from the allocated country
    grading_scheme_added = pd.merge(prim_cleaned_ascents_df,csv_load_countries_df,how ='left', on =['Country'])
    grading_scheme_added = grading_scheme_added.drop('Alpha_3', axis=1)

    # we are adding a new column which will be filled with the consecutive grade
    grading_scheme_added = grading_scheme_added.assign(cons_grade = grading_scheme_added['Route Grade'])
    grading_schemes_in_dataset =[]
    countries_in_dataset =[]

    for index, row in grading_scheme_added.iterrows():#
        # we are getting the grading scheme context and also prepare a list with the other grading schemes.
        all_schemes = list(csv_load_grades_df.columns.values)
        pr_scheme = row['pr_grad_scheme']
        sec_scheme = row['sec_grad_scheme']
        all_schemes.remove(pr_scheme)
        if pd.isna(sec_scheme) == False:
            all_schemes.remove(sec_scheme)

        # We are getting the line with the correct grading scheme. If we can't find it we try the other schemes
        # First we are trying the primary grading scheme
        grade_row = csv_load_grades_df[csv_load_grades_df[pr_scheme]== row['Route Grade']]
        if grade_row.empty:
            # if a secondary scheme is allocated we'll try the secondary scheme
            if pd.isna(sec_scheme) == False:
                grade_row = csv_load_grades_df[csv_load_grades_df[sec_scheme]== row['Route Grade']]

            # If we still don't have a scheme yet we loop through the other grading schemes to look for one
            if grade_row.empty:
                for scheme in all_schemes:
                    grade_row = csv_load_grades_df[csv_load_grades_df[scheme]== row['Route Grade']]
                    if not grade_row.empty:
                        break
    
        # if we still didn't find a grade to allocate we don't have another chance than deleting the ascent (This should only affect very few ascents).
        if grade_row.empty:
            print('Deleted ascent as no grade allocation possible:',row['Route Name'], file =l)
            grading_scheme_added.drop(index, inplace=True)
            continue

        # Now that we're sure, that we found a grade and a grading scheme, we can allocate the found consecutive grade to the dataframe
        #print('Route:',row['Route ID'], ' cons grade:',grade_row['cons_grade'])
        grading_scheme_added.at[index,'cons_grade'] = grade_row['cons_grade'].item()

        # We are also adding the grading scheme to a list, so we can further evaluate a count of schemes in a dataset
        grading_schemes_in_dataset.append(pr_scheme)
        countries_in_dataset.append(row['Country'])

    # Let's count how many occurences of grading_schemes and countries we have -> This is just for data properties
    Count_grading_schemes = Counter(grading_schemes_in_dataset)
    Count_countries = Counter(countries_in_dataset)

    # We convert the cons_grade to int from object
    grading_scheme_added['cons_grade'] = grading_scheme_added['cons_grade'].astype(str).astype(int)

    # We delete a few columns that we got through the merge and don't need anymore.
    # Test with different combinations of cons_grade, Route Grade and Ascent Grade showed the best result for all of it included.
    grading_scheme_added = grading_scheme_added.drop(columns = ['pr_grad_scheme','sec_grad_scheme'])

    #We are dropping the route name, as it is different for each ascent and will only increase the one hot encoding etc.
    grading_scheme_added = grading_scheme_added.drop(columns = ['Route Name'])


    grading_scheme_added = grading_scheme_added.reset_index(drop = True)
    print('After grade adjustment:',grading_scheme_added.shape, file =l)
    return grading_scheme_added, Count_grading_schemes, Count_countries


# We want to write our output into a logfile to keep en overview
log_filename = ('clean_datasets/data_prep_log.txt')
l = open(log_filename,'w')

# We set the variable for the folder with the Coded Data
raw_path = "../Data_collection/Coded_Data"
files = glob.glob(raw_path + "/*.csv")

# We set the path for the cleaned Data
class_clean_path = Path('clean_datasets/class_clean_datasets')
reg_clean_path = Path('clean_datasets/reg_clean_datasets')

# We load the additional files which we will later need for some grade adjustments
csv_load_grades_df = pd.read_csv('../Additional_files/grade_table.csv',delimiter =';')
csv_load_countries_df = pd.read_csv('../Additional_files/Countries.csv',delimiter =';')

# We iterate through all the csv files in the Coded_Data folder and perform our cleaning operations on all of them
for i in range(len(files)):
    csv_load_ascents_df = pd.read_csv(files[i],quotechar = "\"",low_memory=False,delimiter =';',parse_dates=['Ascent Date','Log Date'])
    # This block performs some data cleaning
    cleaned_ascents_df = csv_load_ascents_df.copy()

    # We are only interested in Sport-climbs. Therefore we delete all other gear styles
    # Some datasets are in different languages, therefore we add the french Route Gear Style
    print(files[i],' Initial shape:',cleaned_ascents_df.shape, file =l)
    rel_Route_Gear_Styles = ('Sport','Sportive')
    cleaned_ascents_df = cleaned_ascents_df[cleaned_ascents_df['Route Gear Style'].isin(rel_Route_Gear_Styles)]
    print(files[i],' Shape, after dropping non related climbing styles:',cleaned_ascents_df.shape, file =l)
    
    # We further delete non relevant ascent types
    rel_Ascent_types = ('Onsight','Flash','Red point','Pink point')
    cleaned_ascents_df = cleaned_ascents_df.loc[cleaned_ascents_df['Ascent Type'].isin(rel_Ascent_types)]
    print(files[i],' Shape, after dropping non related climbing methods:',cleaned_ascents_df.shape, file =l)

    # Now we drop some columns, that definitly not have any useful information for the model (e.g. Links, IDs etc.)
    # We also dropp the 'Route Gear Style' as it is only containing the same information now.
    cleaned_ascents_df = cleaned_ascents_df.drop(columns =['Route Link','Country Link','Crag Link','Log Date','Shot','Route ID','Ascent Height','Ascent Label','Ascent Gear Style','Route Gear Style'])

    print(files[i],' Shape, Before dropping nans:',cleaned_ascents_df.shape, file =l)

    # We are changing the stars allocation from stars to a numerical format.
    # If we don't do that, we'd have to get rid of Zero Star routes, as they are a nan value.
    cleaned_ascents_df.loc[cleaned_ascents_df['Route Stars'].isna(), 'Route Stars'] = '0'
    cleaned_ascents_df.loc[cleaned_ascents_df['Route Stars'] == '*', 'Route Stars'] = '1'
    cleaned_ascents_df.loc[cleaned_ascents_df['Route Stars'] == '**', 'Route Stars'] = '2'
    cleaned_ascents_df.loc[cleaned_ascents_df['Route Stars'] == '***', 'Route Stars'] = '3'


    #drop some columns, that MI analysis deemes unworthy
    #cleaned_ascents_df = cleaned_ascents_df.drop('Route Stars',axis =1)
    cleaned_ascents_df = cleaned_ascents_df.drop('# Ascents',axis =1)

    # We have to drop the nan values, as it is not possible to reasonably impute them
    # Option B: Maybe some of the columns with NaN values can be dropped completly, leaving us with more rows?
    cleaned_ascents_df = cleaned_ascents_df.dropna()

    # We'll strip the route heigh of the m (meters) as there are no other measurements used.
    # This makes this into a discrete variable that does not need to be one hot encoded.
    cleaned_ascents_df['Route Height'] = cleaned_ascents_df['Route Height'].map(lambda x: x.rstrip('m'))
    cleaned_ascents_df['Route Height'] = cleaned_ascents_df['Route Height'].astype(str).astype(int)

    # change of data type for the ascent date
    cleaned_ascents_df['Ascent Date'] = pd.to_datetime(cleaned_ascents_df['Ascent Date']).dt.date

    # Some changes only applied to the data that is used for classification task
    # Here we are changing all Onsight to flash, as this is partially based on a decision not the climber skill
    # We also combine Pink / Red point, as these are used mostly interchangably and the use of Pink point went down a lot
    ascents_method_pred_df = cleaned_ascents_df.copy()

    ascents_method_pred_df.loc[ascents_method_pred_df['Ascent Type'] == 'Onsight', 'Ascent Type'] = 'Flash'
    ascents_method_pred_df.loc[ascents_method_pred_df['Ascent Type'] == 'Pink point', 'Ascent Type'] = 'Red point'
    ascents_method_pred_df = ascents_method_pred_df.dropna()

    ascents_method_pred_df = ascents_method_pred_df.reset_index(drop = True)

    # MLP achieved better results without splitting up the date
    ascents_method_pred_df['Ascent_day'] = pd.to_datetime(ascents_method_pred_df['Ascent Date']).dt.day
    ascents_method_pred_df['Ascent_month'] = pd.to_datetime(ascents_method_pred_df['Ascent Date']).dt.month
    ascents_method_pred_df['Ascent_year'] = pd.to_datetime(ascents_method_pred_df['Ascent Date']).dt.year
    ascents_method_pred_df = ascents_method_pred_df.drop('Ascent Date',axis =1)
    
    print(files[i],' Shape, at the end of cleaning:', ascents_method_pred_df.shape,file =l)

    # Now we call the method for grade adjustment
    ascents_grades_adjusted_df, count_grading_schemes, count_countries = consGrade(ascents_method_pred_df)

    # we determine the class distribution and percentage values
    Values_ascent_type = ascents_grades_adjusted_df['Ascent Type'].value_counts()

    print(files[i],' count grading schemes:', count_grading_schemes,file =l)
    print(files[i],' count countries:', count_countries,file =l)
    print(files[i],' Ascent Type value counts',  Values_ascent_type,file =l)

    # We store the cleaned files within an extra folder
    filename = mid(files[i],29, 18)
    filepath = (str(class_clean_path) + filename + '_cleaned.csv')
    
    ascents_grades_adjusted_df.to_csv(filepath,encoding="utf-8",index = False)