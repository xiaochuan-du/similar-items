from numpy import dtype


raw_feature_columns_names = [
    "asin",
    "description",
    "price",
    "imUrl",
    "related",
    "salesRank",
    "categories",
    "title",
    "brand",
]
label_column = "label"

label_column_dtype = {"label": dtype("O")}

raw_feature_columns_dtype = {
    "asin": dtype("O"),
    "description": dtype("O"),
    "price": dtype("float64"),
    "imUrl": dtype("O"),
    "related": dtype("O"),
    "salesRank": dtype("O"),
    "categories": dtype("O"),
    "title": dtype("O"),
    "brand": dtype("O"),
}

feature_columns_dtype = {
    "text": dtype("O"),
}

feature_columns = list(feature_columns_dtype.keys())
output_raw_columns = [label_column] + raw_feature_columns_names
output_engineered_feature_columns = [label_column] + feature_columns
