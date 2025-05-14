"""Data cleaning life_expectancy"""

import pandas as pd
import argparse

def load_data(path: str, delimiter: str = '\t') -> pd.DataFrame:

    loaders = {
        '.csv': lambda p: pd.read_csv(p, sep=','),     
        '.tsv': lambda p: pd.read_csv(p, sep='\t'),    
        '.txt': lambda p: pd.read_csv(p, sep=delimiter),  
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet,
    }

    path = path.lower()
    for file_extension, loader_function in loaders.items():
        if path.endswith(file_extension):
            return loader_function(path)

    raise ValueError(f"Unsupported file format: {path}")




def split_columns(df: pd.DataFrame,
                  column: str,
                  delimiter: str = ',',
                  new_column_names: list[str] = None) -> pd.DataFrame:

    splits = df[column].str.split(delimiter, expand=True)
    num_cols = splits.shape[1]

    if new_column_names:
        if len(new_column_names) != num_cols:
            raise ValueError(f"{len(new_column_names)} names, but {num_cols} columns.")
        new_cols = new_column_names
    else:
        new_cols = [f'col_{i+1}' for i in range(num_cols)]

    df[new_cols] = splits
    df.drop(columns=[column], inplace=True)

    return df



def unpivot(df: pd.DataFrame, exclude_columns: list,
            identifier_col_name="identifier", value_col_name="value") -> pd.DataFrame:


    value_columns = [col for col in df.columns if col not in exclude_columns]


    unpivoted_df = pd.melt(df,
                           id_vars=exclude_columns,
                           value_vars=value_columns,
                           var_name=identifier_col_name,
                           value_name=value_col_name)


    return unpivoted_df



def clean_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:

    df[column] = df[column].astype(str).str.strip()

    symbols_to_remove = ['$', ',', '%', '€', ':', 'kg', 'approx.', 'N/A', 'missing']
    for sym in symbols_to_remove:
        df[column] = df[column].str.replace(sym, '', regex=False)


    df[column] = df[column].str.extract(r'(-?\d+\.?\d*)', expand=False)

    df = df.dropna(subset=[column])

    return df



def convert_column_dtype(df, column_name, target_dtype):

    df[column_name] = df[column_name].astype(target_dtype)
    return df



def filter_rows(df: pd.DataFrame, column: str, value) -> pd.DataFrame:

    return df[df[column] == value]



def save_dataframe(df: pd.DataFrame, filename) -> None:
    filename = str(filename)
    
    if filename.endswith('.csv'):
        df.to_csv(filename, index=False)
    elif filename.endswith('.xlsx'):
        df.to_excel(filename, index=False)
    elif filename.endswith('.json'):
        df.to_json(filename, orient='records', lines=True)
    elif filename.endswith('.parquet'):
        df.to_parquet(filename, index=False)
    else:
        raise ValueError(f"Unsupported file format for: {filename}")


def clean_data(input_path: str, output_path: str, country: str = "PT"):

    df = load_data(input_path)

    id_columns = ['unit', 'sex', 'age', 'region']

    df = split_columns(df, column='unit,sex,age,geo\\time',
                       delimiter=',', new_column_names=id_columns)


    df = unpivot(df, exclude_columns=id_columns,
                 identifier_col_name='year', value_col_name='value')

    df = clean_numeric_column(df, 'year')
    df = convert_column_dtype(df, 'year', int)

    df = clean_numeric_column(df, 'value')
    df = convert_column_dtype(df, 'value', float)

    df = filter_rows(df, 'region', country)

    save_dataframe(df, output_path)


if __name__ == "__main__":  # pragma: no cover


    parser = argparse.ArgumentParser(description="Clean EU life expectancy data.")
    parser.add_argument("input_path", help="Path to input data file")
    parser.add_argument("output_path", help="Path to save cleaned data")
    parser.add_argument("--country", default="PT", help="Country code to filter (default: PT)")

    args = parser.parse_args()

    clean_data(
        input_path=args.input_path,
        output_path=args.output_path,
        country=args.country
    )
