import pandas as pd
import numpy as np
import multiprocessing

#================ Parallel functions start =====================

def samples_count(chunk):
    '''
    Returns the number of samples in a chunk.
    This function is called using multiple chunks in parallel.
    '''
    return len(chunk)


def numeric_summary(chunk, cols_to_use):
    '''
    Returns the max, min and approx average of numeric columns in a chunk.
    This function is called using multiple chunks in parallel.
    '''
    sample = chunk[cols_to_use].dropna() # selecting float cols
    mins_ = sample.apply(min, axis=0).values # apply min function to all cols in df
    maxs_ = sample.apply(max, axis=0).values # apply max function to all cols in df
    means_ = sample.apply(np.mean, axis=0).values # apply mean function to all cols in df
    return (mins_, maxs_, means_)


def unique_cat_values(chunk, cols_to_use):
    '''
    Returns the unique values of categorical columns in a chunk.
    This function is called using multiple chunks in parallel.
    '''
    sample = chunk[cols_to_use].copy()
    sample.fillna("missing_value", inplace=True)
    unique_ = sample.apply(np.unique).to_dict() # apply np.unique to cols and converting to dict
    return unique_


def find_value_counts(chunk, cols_to_use):
    '''
    Returns the value counts of categories in categorical columns of a chunk.
    This function is called using multiple chunks in parallel.
    '''
    sample = chunk[cols_to_use].copy()
    value_counts = dict()
    sample.fillna("missing_value", inplace=True)
    for col in cols_to_use:
        value_counts[col] = sample[col].value_counts().to_dict() # for first chunk storing just initial val count
    return value_counts


def find_missing_value_counts(chunk, cols_to_use):
    '''
    Returns the misisng value count of all columns in a chunk.
    This function is called using multiple chunks in parallel.
    '''
    sample = chunk[cols_to_use].copy()
    missing_value_counts = sample.isna().sum()
    return missing_value_counts

#================ Parallel functions End =====================

# ============== Entry points Start ==========================

def get_data_shape(chunks):
    '''
    Get the data shape by parallely calculating lenght of each chunk and 
    aggregating them to get lenght of complete training dataset
    '''
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    n_cols = 0
    for n, chunk in enumerate(chunks):
        if n == 0:
            n_cols = chunk.shape[1] # storing ncols of first chunk
        f = pool.apply_async(samples_count, [chunk]) # asynchronously applying function to chunk. Each worker parallely begins to work on the job
        results.append(f) # appending result to results
        
    n_samples = 0
    for f in results:
        n_samples += f.get(timeout=120) # getting output of each parallel job
        
    pool.close()
    pool.join()
    return n_samples, n_cols


def get_numeric_summary(chunks, cols_to_use):
    '''
    Compute min and max values of each chunk in parallel then find the global min and max.
    Compute mean of chunk means to get an approx mean value of the data
    '''
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    for chunk in chunks:
        f = pool.apply_async(numeric_summary, args=(chunk, cols_to_use)) # asynchronously applying function to chunk. Each worker parallely begins to work on the job
        results.append(f) # appending result to results
        
    
    for n, f in enumerate(results):
        res = f.get(timeout=120) # getting output of each parallel job
        if n == 0:
            mins_ = res[0] # initializing mins_
            maxs_ = res[1] # initializing maxs_
            approx_means_ = res[2] # initializing means_
        else:
            mins_ = np.vstack([mins_, res[0]]) # vstacking all mins_
            maxs_ = np.vstack([maxs_, res[1]]) # vstacking all maxs_
            approx_means_ = np.vstack([approx_means_, res[2]]) # vstacking all means_

    mins_ = np.min(mins_, axis=0) # computing Grand min of rows (axis=0) in mins_
    maxs_ = np.max(maxs_, axis=0) # computing Grand max of rows (axis=0) in maxs_
    approx_means_ = np.mean(approx_means_, axis=0) # computing Grand mean of rows (axis=0) in means_
    
    pool.close()
    pool.join()
    return mins_, maxs_, approx_means_


def get_value_counts_prop(chunks, cols_to_use):
    '''
    Get value counts and proportions of categorical variables by parallely applying find_value_counts function
    '''
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    for chunk in chunks:
        f = pool.apply_async(find_value_counts, args=(chunk, cols_to_use)) # asynchronously applying function to chunk
        results.append(f) # appending result to results
        
        
    value_counts = dict()
    value_counts_prop = dict()
    for n, f in enumerate(results):
        res = f.get(timeout=120) # getting output of each parallel job
        for col in cols_to_use:
            if n == 0:
                value_counts[col] = res[col] # for first chunk storing just initial val count
            else:
                previous_counts = value_counts[col] # previous value counts for col
                current_counts = res[col] # current value counts for col
                # updating val count dict of each col after adding current val counts to prev val counts 
                value_counts[col].update({key: (previous_counts[key] if previous_counts.get(key) is not None else 0) \
                                                + (current_counts[key] if current_counts.get(key) is not None else 0) \
                                          for key in current_counts.keys()})
                
    pool.close()
    pool.join()
    for col in cols_to_use:
        value_counts[col] = {key: value for key, value in sorted(value_counts[col].items(), key=lambda x: x[1], reverse=True)}
        col_sum = sum(value_counts[col].values())
        value_counts_prop[col] = {key: np.round(value / col_sum, 6) for key, value in value_counts[col].items()}
        
    
    return value_counts, value_counts_prop


def get_missing_value_counts_prop(chunks, cols_to_use):
    '''
    Get missing value counts and proportions of all variables by parallely applying find_missing_value_counts function
    '''
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    n_samples = 0
    for chunk in chunks:
        n_samples += len(chunk)
        f = pool.apply_async(find_missing_value_counts, args=(chunk, cols_to_use)) # asynchronously applying function to chunk
        results.append(f) # appending result to results
             
    for n, f in enumerate(results):
        res = f.get(timeout=120) # getting output of each parallel job
        if n == 0:
            missing_value_counts = res # for first chunk storing just initial val count
        else:
            previous_counts = missing_value_counts # previous missing value counts
            current_counts = res # current missing value counts
            # updating val count dict of each col after adding current val counts to prev val counts 
            missing_value_counts = previous_counts + current_counts
                
    pool.close()
    pool.join()
    
    missing_value_counts.sort_values(inplace=True, ascending=False)
    missing_value_prop = missing_value_counts / n_samples 
    
    return missing_value_counts, missing_value_prop

# ============== Entry points End ==========================