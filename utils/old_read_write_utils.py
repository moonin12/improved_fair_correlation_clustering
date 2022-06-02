'''
From https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
Slightly modified in places
'''


# Clean the data. Bucketize text data, convert int to float.
# Arguments:
#   df (pd.DataFrame) : DataFrame containing the data
#   config (ConfigParser) : Config object containing the specifications of
#       which columns are text.
#   dataset (str) : Name of dataset being used.
def clean_data(df, numeric_features, text_features):
    # Remove the unnecessary columns. Save the variable of interest column, in case
    selected_columns = numeric_features + text_features
    df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    offset = 0
    for col in df:
        if col in text_features:
            # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
            df[col] = df[col].astype('category').cat.codes
            df[col] = df[col] + offset
            offset += len(df[col].unique())
        if col in numeric_features:
            df[col] = df[col].astype(float)
            # Scale to [0,1]
            max_val = df[col].max()
            min_val = df[col].min()
            df[col] = (df[col] - min_val) / (max_val - min_val)

    return df
