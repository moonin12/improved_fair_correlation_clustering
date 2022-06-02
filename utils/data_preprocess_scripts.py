# Script to read raw csv files for bank, cnesus, and diabetes, clean them up, and subsample

from utils.read_write_utils import read_csv_data, write_csv_data

sample_size = 200
dataset_names = ["bank", "census", "diabetes"]
separator = {"bank": ';', "census": ',', "diabetes": ','}
inoutdirectory = "data/"
numerics = {"bank": ["age", "balance", "duration"],
            "census": ["age", "final-weight", "education-num", "capital-gain", "hours-per-week"],
            "diabetes": ["age", "time_in_hospital"]
            }
texts = {"bank": ["marital", "default"],
         "census": ["sex", "race"],
         "diabetes": ["race", "gender"]}

# for dataset_name in dataset_names:
#     df = read_csv_data(dataset_name + "_raw", inoutdirectory, has_index=False, separator=separator[dataset_name])
#     df = clean_data(df, numerics[dataset_name], texts[dataset_name])
#     write_csv_data(dataset_name, inoutdirectory, df)

for dataset_name in dataset_names:
    df = read_csv_data(dataset_name, inoutdirectory, has_index=False)
    fraction = sample_size / df.shape[0]
    df = df.groupby(texts[dataset_name]).apply(lambda x: x.sample(frac=fraction))
    write_csv_data(dataset_name + "_" + str(sample_size), inoutdirectory, df)
