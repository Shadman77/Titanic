import pandas as pd
import sys

#get sir name
def getSirName(df):
    df['sir_name'] = 'no_sir_name'
    for i in range(len(df.index)):
        tmpList = df.loc[i]['Name'].split(",")
        if len(tmpList) > 0:
            df['sir_name'][i] = tmpList[0].strip()

    return df

#get name prefix
def getNamePrefix(df):
    df['name_prefix'] = 'no_name_prefix'
    for i in range(len(df.index)):
        try:
            df['name_prefix'][i] = df.loc[i]['Name'].split('.')[0].split(',')[1].strip()
        except:
            pass
    return df

# find age for missing values
def avgAge(df, df_mean):
    for i in range(len(df.index)):
        if pd.isna(df.loc[i]['Age']):
            if df.loc[i]['Parch'] > 0:
                mean = df_mean[(df_mean['name_prefix'] == df.loc[i]['name_prefix']) & df_mean['Parch'] > 0]['Age'].mean(skipna = True)
            else:
                mean = df_mean[(df_mean['name_prefix'] == df.loc[i]['name_prefix']) & df_mean['Parch'] == 0]['Age'].mean(skipna = True)
            df['Age'][i] = mean
            print(df.loc[i]['Age'])
    # if there are any nan values
    df['Age'] = df['Age'].fillna(df_mean['Age'].mean(skipna = True))
    return df

#convert pclass to a categorical column
def pclassToCat(df):
    for i in range(len(df.index)):
        if df.loc[i]['Pclass'] == 1:
            df['Pclass'][i] = 'First_Class'
        elif df.loc[i]['Pclass'] == 2:
             df['Pclass'][i] = 'Second_Class'
        elif df.loc[i]['Pclass'] == 3:
             df['Pclass'][i] = 'Third_Class'
    return df


# clean the data and remove unwanted columns
def data_clean(df, df_mean):
    # show info
    print(df.info())

    print('Unique Values')
    print(len(df['Ticket'].unique()))

    #get sir name
    df = getSirName(df)

    #get name prefix
    df = getNamePrefix(df)
    df_mean = getNamePrefix(df_mean)

    # fill nas age
    df = avgAge(df, df_mean)
    #df['Age'] = df['Age'].fillna(0)#FILL THIS WITH THE MEAN VALUE???????
    #df = df.dropna()

    #convert pclass into a categorical column
    #df = pclassToCat(df) #decreased accuracy

    # drop unwanted columns
    df = df.drop([
        'Name'], axis=1)

    # show info
    print(df.info())


    return df


def main():
    print('Load Data')
    raw_dataset = pd.read_csv(
        'data/train.csv', skipinitialspace=True, verbose=True)
    train_dataset = raw_dataset.copy()
    raw_dataset = pd.read_csv(
        'data/test.csv', skipinitialspace=True, verbose=True)
    test_dataset = raw_dataset.copy()
    print(len(train_dataset.index))
    print(len(test_dataset.index))

    #mean dataset
    mean_dataset = pd.concat([train_dataset, test_dataset])
    mean_dataset = mean_dataset.drop(['Survived'], axis=1)

    # clean the data and remove unwanted columns
    train_dataset = data_clean(train_dataset, mean_dataset)
    test_dataset = data_clean(test_dataset, mean_dataset)

    #save the datasets
    train_dataset.to_csv('data/cleaned_train.csv', index=False)
    test_dataset.to_csv('data/cleaned_test.csv', index=False)


if __name__ == "__main__":
    main()