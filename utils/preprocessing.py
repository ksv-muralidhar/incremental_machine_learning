import numpy as np
import pandas as pd
import math
from datetime import datetime
from sklearn.utils import class_weight
import multiprocessing
from sklearn.decomposition import IncrementalPCA
import cloudpickle


#================ Parallel functions start =====================

def clean_size_column(chunk):
    '''
    Cleans the size column by removing ',' and converting it into integer.
    This function is parallely called with each chunk.
    '''
    def clean_size(s):
        if type(s) == str:
            s = s.lower()
            s = "".join(s.split())
            s = s.replace(",", "")
            if s.endswith("k"):
                s = float(s.replace("k", ""))
                s = int(s * 10 ** 3)
            elif s.endswith("m"):
                s = float(s.replace("m", ""))
                s = int(s * 10 ** 6)
            elif s.endswith("g"):
                s = float(s.replace("g", ""))
                s = int(s * 10 ** 9)
            else:
                s = math.nan
        return s
    
    return chunk['Size'].map(clean_size)


def clean_installs_column(chunk):
    '''
    Cleans the installs column by removing ',' and '+' and converting it into integer.
    This function is parallely called with each chunk.
    '''

    def clean_installs(s):
        if type(s) == str:
            s = s.lower()
            s = "".join(s.split())
            s = s.replace(",", "")
            s = s.replace("+", "")
            s = int(s)
        return s
    
    return chunk['Installs'].map(clean_installs)


def drop_unwanted_cols(chunk, cols_to_drop):
    '''
    Drops unwanted columns from a chunk
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    chunk.drop(columns=cols_to_drop, inplace=True)
    return chunk


def encode_boolean_cols(chunk, cols_to_encode):
    '''
    Maps boolean columns with integers. True = 1, False = 0.
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    bool_map = {True: 1, False: 0}
    for col in cols_to_encode:
        chunk[col] = chunk[col].map(bool_map)
    return chunk


def convert_date_to_days(chunk, date_cols_to_convert):
    '''
    Computes the date difference betwwen today and a given date
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    def convert_date(date):
        try:
            if type(date) is str:
                return (datetime.today().date() - datetime.strptime(date, "%b %d, %Y").date()).days
            return date
        except:
            return math.nan
    
    for col in date_cols_to_convert:
        chunk[col] = chunk[col].map(convert_date)
    return chunk


def missing_value_imputer(chunk, numeric_cols_to_impute: list, numeric_values_to_impute: list,
                          categorical_cols_to_impute: list, categorical_values_to_impute: list):
    '''
    Imputes missing values in the specified columns with specified values
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    for col, impute_value in zip(numeric_cols_to_impute, numeric_values_to_impute):
        chunk[col] = chunk[col].fillna(impute_value)
        
    for col, impute_value in zip(categorical_cols_to_impute, categorical_values_to_impute):
        chunk[col] = chunk[col].fillna(impute_value)
        
    return chunk


def rare_category_encoder(chunk, frequent_categories):
    '''
    Replaces rare categories in categorical features with 'rare_category'
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    for col in frequent_categories:
        chunk.loc[chunk[col].isin(frequent_categories[col]) == False, col] = f"{col}_rare_category"
    
    return chunk


def one_hot_encoder(chunk, one_hot_encode_categories: dict):
    '''
    One-hot-encodes categorical columns.
    This function is parallely called with each chunk.
    '''

    chunk = chunk.copy()
    for col in one_hot_encode_categories:
        for cat in one_hot_encode_categories[col]:
            chunk[f'{col}_{cat}'] = 0
            chunk.loc[chunk[col] == cat, f'{col}_{cat}'] = 1
        
    chunk.drop(columns=[*one_hot_encode_categories.keys()], inplace=True)
    
    return chunk


def preprocess_data(chunk, bool_cols_to_encode, date_cols_to_convert, 
                    numeric_cols_to_impute: list, numeric_values_to_impute: list,
                    categorical_cols_to_impute: list, categorical_values_to_impute: list,
                    frequent_categories: dict, one_hot_encode_categories: dict):
    
    '''
    Pipeline to apply all the preprocessing tasks to each chunk.
    This function is parallely called with each chunk.
    '''
    chunk = chunk.copy()
    chunk['Size'] = clean_size_column(chunk)
    chunk['Installs'] = clean_installs_column(chunk)
    chunk = encode_boolean_cols(chunk, bool_cols_to_encode)
    chunk = convert_date_to_days(chunk, date_cols_to_convert)
    chunk = missing_value_imputer(chunk, numeric_cols_to_impute, numeric_values_to_impute,
                            categorical_cols_to_impute, categorical_values_to_impute)
    chunk = rare_category_encoder(chunk, frequent_categories)
    chunk = one_hot_encoder(chunk, one_hot_encode_categories)
    
    return chunk


def minmax_scaler(chunk, mins, maxs):
    '''
    Performs minmax scaling on each chunk.
    This function is parallely called with each chunk.
    '''
    x = chunk.copy()
    for col, min_, max_ in zip([*x.columns], mins, maxs):
        x[col] = (x[col] - min_) / (max_ - min_)
    return x


#================ Parallel functions end =====================

# ============== Entry points Start ==========================

def get_and_save_label_encoding_and_class_weights(y_train, only_labels, class_weights_save_path, encoded_y_save_path):
    '''
    Encodes labels (y), computes class weights and saves them
    '''
    y = np.array([])
    for chunk in y_train:
        y = np.append(y, chunk.values.ravel()) # ravel chunk.values as chunk.values is 2D array because chunk is a df with 1 column
    
    label_map = {0.0: 0, 1.0: 1, 1.1: 2, 1.2: 3, 1.3: 4, 1.4: 5, 1.5: 6, 1.6: 7, 1.7: 8, 1.8: 9, 1.9: 10, 2.0: 11, 2.1: 12, 
                2.2: 13, 2.3: 14, 2.4: 15, 2.5: 16, 2.6: 17, 2.7: 18, 2.8: 19, 2.9: 20, 3.0: 21, 3.1: 22, 3.2: 23,
                3.3: 24, 3.4: 25, 3.5: 26, 3.6: 27, 3.7: 28, 3.8: 29, 3.9: 30, 4.0: 31, 4.1: 32, 4.2: 33, 4.3: 34, 4.4: 35, 4.5: 36, 
                4.6: 37, 4.7: 38, 4.8: 39, 4.9: 40, 5.0: 41}
    
    y = pd.Series(y)
    y = y.map(label_map)
    
    y.to_csv(encoded_y_save_path, index=False)
    
    if only_labels == False:
        classes_ = sorted([*y.unique()])  # returns sorted unique class labels
        # Dictionary comprehension that assigns each class (key) its weight (value)
        class_weights = {class_: weight for class_, weight in
                              zip(classes_, class_weight.compute_class_weight(class_weight='balanced',
                                                                              classes=classes_, y=y))}
        
        with open(class_weights_save_path, "wb") as f:
            cloudpickle.dump(class_weights, f)
    


def del_unwanted_columns(x):
    '''
    Drop unwanted columns by parallely applying drop_unwanted_cols function
    '''
    # nested function to parallelize drop_unwanted_cols function in utils
    def get_wanted_columns(chunks, cols_to_drop):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = []
        for chunk in chunks:
            f = pool.apply_async(drop_unwanted_cols, args=(chunk, cols_to_drop)) # asynchronously applying function to chunk
            results.append(f) # appending result to results

        for n, f in enumerate(results):
            if n == 0: 
                pool.close()
                pool.join()
            yield f.get(timeout=120) # getting output of each parallel job
        
    # calling nested function to get data with only wanted cols
    cols_to_drop = ['App Name', 'App Id', 'Minimum Android', 'Developer Id', 'Developer Website', 'Developer Email', 'Privacy Policy', 'Scraped Time']
    x_with_wanted_cols = get_wanted_columns(x, cols_to_drop)
    return x_with_wanted_cols


def get_preprocessed_data(x):
    '''
    Preprocesses data by parallely applying preprocess_data function
    '''
    # nested function to parallelize preprocess_data function in utils
    def return_preprocessed_data(x, bool_cols_to_encode, date_cols_to_convert, 
                              numeric_cols_to_impute, numeric_values_to_impute,
                              categorical_cols_to_impute, categorical_values_to_impute,
                              frequent_categories, one_hot_encode_categories):
    
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = []
        for chunk in x:
            f = pool.apply_async(preprocess_data, args=(chunk, bool_cols_to_encode, date_cols_to_convert,
                                                        numeric_cols_to_impute, numeric_values_to_impute,
                                                        categorical_cols_to_impute, categorical_values_to_impute,
                                                        frequent_categories, one_hot_encode_categories)) # asynchronously applying function to chunk
            results.append(f) # appending result to results

        for n, f in enumerate(results):
            if n == 0:
                pool.close()
                pool.join()
            yield f.get(timeout=120) # getting output of each parallel job
        
    # calling nested function to get preprocessed data    
    bool_cols_to_encode = ['Free', 'Ad Supported',	'In App Purchases',	'Editors Choice']
    date_cols_to_convert = ['Released', 'Last Updated']
    numeric_cols_to_impute = ['Rating Count', 'Installs', 'Minimum Installs',
                              'Maximum Installs', 'Price', 'Size', 'Released', 'Last Updated']
    numeric_values_to_impute = [2.83122633e+03, 1.85272023e+05, 1.85272023e+05,
                                3.31805629e+05, 1.03541016e-01, 1.92013715e+07, 1.86694232e+03,
                                1.40332171e+03]
    categorical_cols_to_impute = ['Category', 'Free', 'Currency', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice']
    categorical_values_to_impute = ['Education', 1, 'USD', 'Everyone', 0, 0, 0]


    frequent_categories = {"Category": ['Education', 'Music & Audio', 'Business', 'Tools', 'Entertainment', 'Lifestyle', 
                           'Books & Reference', 'Personalization','Health & Fitness', 'Productivity', 
                           'Shopping', 'Food & Drink'], 
                           "Currency": ['USD', 'XXX', 'EUR'],
                           'Content Rating': ['Everyone', 'Teen', 'Mature 17+']}

    one_hot_encode_categories = {'Category': ['Books & Reference', 'Business', 'Category_rare_category',
                                              'Education', 'Entertainment', 'Food & Drink', 'Health & Fitness',
                                              'Lifestyle', 'Music & Audio', 'Personalization', 'Productivity',
                                              'Shopping', 'Tools'],
                                 'Currency': ['Currency_rare_category', 'EUR', 'USD', 'XXX'],
                                 'Content Rating': ['Content Rating_rare_category', 'Everyone', 'Mature 17+', 'Teen']}


    preprocessed_x = return_preprocessed_data(x=x, 
                                                 bool_cols_to_encode=bool_cols_to_encode,
                                                 date_cols_to_convert=date_cols_to_convert,
                                                 numeric_cols_to_impute=numeric_cols_to_impute, 
                                                 numeric_values_to_impute=numeric_values_to_impute,
                                                 categorical_cols_to_impute=categorical_cols_to_impute,
                                                 categorical_values_to_impute=categorical_values_to_impute,
                                                 frequent_categories=frequent_categories,
                                                 one_hot_encode_categories=one_hot_encode_categories)
    
    return preprocessed_x


def get_scaled_data(x):
    '''
    Scales numerics columns by parallely applying minmax_scaler function
    '''
    # nested function to parallelize minmax_scaler function in utils
    def scale_data(x, mins, maxs):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = []
        for chunk in x:
            f = pool.apply_async(minmax_scaler, args=(chunk, mins, maxs)) # asynchronously applying function to chunk
            results.append(f) # appending result to results

        for n, f in enumerate(results):
            if n == 0:
                pool.close()
                pool.join()
            yield f.get(timeout=120) # getting output of each parallel job
        
    # calling nested function to get scaled data
    mins = [  0.,    0.,    0.,    0.,    0.,    0., 3300.,  847.,  846.,
              0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
              0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
              0.,    0.,    0.,    0.,    0.,    0.]

    maxs = [1.3855757e+08, 1.0000000e+10, 1.0000000e+10, 1.2057627e+10,
           1.0000000e+00, 3.9999000e+02, 1.5000000e+09, 5.0030000e+03,
           5.3560000e+03, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
           1.0000000e+00]

    scaled_x = scale_data(x, mins=mins, maxs=maxs)
    return scaled_x


def save_preprocessed_data(x, save_path):
    '''
    Incrementally saves a given dataset by appending chunks to a csv file.
    '''
    try:
        success = 0
        first_chunk = True
        for chunk in x:
            if isinstance(chunk, np.ndarray):
                chunk = pd.DataFrame(chunk)
            if first_chunk == True :
                chunk.to_csv(save_path, index=False)
                first_chunk = False
            else:
                chunk.to_csv(save_path, mode="a", header=False, index=False)
    except:
        success = 0
        raise
    else:
        success = 1
    return success


def get_saved_preprocessed_data(path, chunksize):
    '''
    Reads a CSV file and returns the chunks generator.
    '''
    preprocessed_x = pd.read_csv(path, chunksize=chunksize)
    return preprocessed_x


def fit_and_save_incremental_pca(chunks, model_save_path):
    '''
    Incrementally fits Incremental PCa using partial_fit and saves the PCA model
    '''
    inc_pca = IncrementalPCA(n_components=16)
    for chunk in chunks:
        inc_pca.partial_fit(chunk)

    with open(model_save_path, "wb") as f:
        cloudpickle.dump(inc_pca, f)

    return inc_pca
    
    
def get_principal_components(preprocessed_x, inc_pca):
    '''
    Uses a Pre-fitted PCA model to transform the data into principal components.
    '''
    for x in preprocessed_x:
        x = inc_pca.transform(x)
        yield x
        
    
def get_x_y(x_path, y_path, chunksize):
    '''
    Reads X and y sets from specified paths and returns the chunks generator 
    '''
    x = pd.read_csv(x_path, chunksize=chunksize)
    y = pd.read_csv(y_path, chunksize=chunksize)
    return x, y


def data_preprocess_pipeline(x_path, y_path, preprocessed_x_save_path, encoded_y_save_path, pca_data_save_path, 
                             pca_model_save_path, class_weights_save_path, chunksize, dataset_type):
    
    '''
    Data Preprocessing pipeline to ingest the data and return princial components.
    '''
    
    print('Entering get_x_y')
    x, y = get_x_y(x_path=x_path, y_path=y_path, chunksize=chunksize)
    print('Exiting get_x_y')
    
    
    if dataset_type == 'train':
        print('Entering get and save class weights and label encoding')
        get_and_save_label_encoding_and_class_weights(y, only_labels=False, class_weights_save_path=class_weights_save_path,
                                                      encoded_y_save_path=encoded_y_save_path)
        print('Exiting get and save class weights and label encoding')        
    else:
        print('Entering get and save label encoding')
        get_and_save_label_encoding_and_class_weights(y, only_labels=True, class_weights_save_path=None, encoded_y_save_path=encoded_y_save_path)
        print('Exiting get and save label encoding')
    
    
    # getting training data again as the above generator is exhausted
    print('Entering get_x')
    x, _ = get_x_y(x_path=x_path, y_path=y_path, chunksize=chunksize)
    print('Exiting get_x')
    
    print('Entering del_unwanted_columns')
    x = del_unwanted_columns(x)
    print('Exiting del_unwanted_columns')
    
    print('Entering get_preprocessed_data')
    x = get_preprocessed_data(x)
    print('Exiting get_preprocessed_data')
    
    
    print('Entering get_scaled_data')
    x = get_scaled_data(x)
    print('Exiting get_scaled_data')
    
    
    # save preprocessed data
    print('Entering save_preprocessed_data')
    _ = save_preprocessed_data(x, save_path=preprocessed_x_save_path)
    print('Exiting save_preprocessed_data')
    
    
    inc_pca = None
    if dataset_type == 'train':    
        # read the saved preprocessed data
        print('Entering get_saved_preprocessed_data')
        x = get_saved_preprocessed_data(path=preprocessed_x_save_path, chunksize=chunksize)
        print('Exiting get_saved_preprocessed_data')

        # fitting Inc PCA
        print('Entering fit_and_save_incremental_pca')
        inc_pca = fit_and_save_incremental_pca(x, pca_model_save_path)
        print('Exiting fit_and_save_incremental_pca')
    else:
        print('Loading prefitted incremental PCA model')
        with open(pca_model_save_path, "rb") as f:
            inc_pca = cloudpickle.load(f)
        print('Loaded prefitted incremental PCA model')
    
    # executing the read preprocessing step again, since the above preprocess generator is exhausted
    print('Entering get_saved_preprocessed_data')
    x = get_saved_preprocessed_data(path=preprocessed_x_save_path, chunksize=chunksize)
    print('Exiting get_saved_preprocessed_data')
    
    # getting principal components
    print('Entering get_principal_components')
    x = get_principal_components(x, inc_pca)
    print('Exiting get_principal_components')
    
    
    # save principal components
    print('Entering save principal components')
    _ = save_preprocessed_data(x, save_path=pca_data_save_path)
    print('Exiting save principal components')

# ============== Entry points End ==========================