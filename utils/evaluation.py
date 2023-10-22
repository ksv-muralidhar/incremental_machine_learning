import cloudpickle
from sklearn.metrics import roc_auc_score


def load_model(model_path):
    '''
    Loads the saved prefitted SGDClassifier
    '''
    with open(model_path, 'rb') as f:
        model = cloudpickle.load(f)
    return model


def predict(x, y, model):
    '''
    Generator to return predictions of each chunk. Returns a tuple of y_true and y_pred
    '''
    for x_, y_ in zip(x, y):
        y_ = y_.iloc[:, 0].copy()
        pred = model.predict_proba(x_)
        yield y_, pred
        
        
def model_evaluation(chunks):
    '''
    Returns the mean AUC ROC across the predictions made for each chunk
    '''
    roc_auc = -1
    for chunk in chunks:
        y_true = chunk[0]
        y_pred = chunk[1]
        if roc_auc == -1:
            roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
        else:
            roc_auc = (roc_auc + roc_auc_score(y_true, y_pred, multi_class="ovr")) / 2
    return roc_auc
