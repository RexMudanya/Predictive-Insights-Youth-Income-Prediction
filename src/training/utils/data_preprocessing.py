import pandas as pd


def preprocess_data(data) -> pd.DataFrame:
    df = pd.read_csv(data) if type(data) == str else data

    df['Year_survey'] = pd.to_datetime(df['Survey_date']).dt.year
    df['Age_survey'] = df['Year_survey'] - df['Birthyear']

    # Create a list of categorical variables
    selected_vars = ["Round", "Status", "Geography", "Province",
                     "Schoolquintile", "Math", "Mathlit", "Additional_lang", "Home_lang", "Science"]
    # Remove variables we will not use
    df_dummy = df.drop(["Person_id", "Survey_date"], axis=1)

    # Convert character variables to dummy variables
    df_dummy = pd.get_dummies(df_dummy, columns=selected_vars, drop_first=True, dummy_na=True)

    # Clean column names
    df_dummy.columns = df_dummy.columns.str.replace(' ', '_')  # Replace spaces with underscores
    df_dummy.columns = df_dummy.columns.str.replace('[^\w\s]', '', regex=True)  # Remove special characters
    df_dummy.columns = df_dummy.columns.str.replace('_+', '_',
                                                    regex=True)  # Replace consecutive underscores with a
    # single underscore
    df_dummy.columns = df_dummy.columns.str.rstrip('_')  # Remove trailing underscores at the end

    df_dummy = df_dummy.fillna(0)

    return df_dummy
