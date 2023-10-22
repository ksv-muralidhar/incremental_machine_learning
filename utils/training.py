from sklearn.linear_model import SGDClassifier
import cloudpickle


def load_class_weights(class_weights_path):
    '''
    loads the saved class weights from a specified path.
    '''
    with open(class_weights_path, 'rb') as f:
        class_weights = cloudpickle.load(f)
    return class_weights


def fit_and_save_sgd_classifier(x_train, y_train, class_weights, model_save_path):
    '''
    Incrementally fits SGDClassifier to data chunks using partial_fit
    '''
    print('Entering fit_and_save_sgd_classifier')
    sgd = SGDClassifier(class_weight=class_weights, random_state=42, loss="log_loss")
    first_iter = True
    for x, y in zip(x_train, y_train):
        y = y.iloc[:, 0].copy()
        if first_iter:
            sgd.partial_fit(x, y, classes=[*class_weights.keys()])
            first_iter = False
        else:
            sgd.partial_fit(x, y)
    
    with open(model_save_path, 'wb') as f:
        cloudpickle.dump(sgd, f)
    
    print('Saved the Model & Exiting fit_and_save_sgd_classifier')
