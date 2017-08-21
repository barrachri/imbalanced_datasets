import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc

SIZE_IMAGE = (10,10)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.
    The returned matrix is normalized but shows the original value
    """
    
    plt.figure(figsize=SIZE_IMAGE)
    
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_n, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_n.max() / 3.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        color="white" if cm_n[i, j] > thresh else "black"
        
        value_normalized = f"{cm_n[i, j]:.2f}"
        value = f"({cm[i, j]})" 
        
        # print original values
        plt.text(j, i, format(value),
                 horizontalalignment="center",
                 verticalalignment='top',
                 color=color)
        
        # print normalized values
        plt.text(j, i, format(value_normalized),
                 horizontalalignment="center",
                 verticalalignment='bottom',
                 color=color)  
          

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
def plot_score_curve(title, scores, ylim=None):
    """
    Plot a collection a scores.
    
    scores: an np.array
    """
    plt.figure(figsize=SIZE_IMAGE)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iteration")
    plt.ylabel("Score")

    test_scores_mean = scores.mean()
    test_scores_std = scores.std()
    
    plt.grid()

    plt.fill_between(range(len(scores)), scores+test_scores_std, scores-test_scores_std, alpha=0.2, color="r")

    plt.plot(scores, '-', color="g",
             label="F1-Score")
    
    plt.plot([scores.mean()]*len(scores), '--', color="b",
             label="F1-Score mean")

    plt.legend(loc="best")
    plt.show()

from sklearn.model_selection import RepeatedKFold


def clean_data(train, test):
    """
    Clean our dataset.
    Remove `Churn` and `Phone` columns from x_* dataset
    and return a y_* dataset with the label
    """
    y_train = train['Churn']
    X_train = train.drop(["Churn","Phone"], axis=1)
    
    y_test = test['Churn']
    X_test = test.drop(["Churn","Phone"], axis=1)
    return X_train, y_train, X_test, y_test


def random_under_sampling(data, n_splits, n_repeats):
    """
    Random under sampling function
    Divide the data in `n_splits` samples
    and repeat the process `n_repeats` times.
    
    data: original data set
    n_splits: number of splits for RepeatedKFold
    n_repeats: number of repetition
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    
    # train and test are a collection of indexes from data
    for train, test in rkf.split(data):
        
        # locate the training samples
        train_samples = data.iloc[train]
        
        number_of_samples = train_samples.shape[0]
        
        # dataset from my fold with Churn == 0
        negative_samples = train_samples.query('Churn == 0')
        # n. rows inside my negative_set
        len_negative_samples = negative_samples.shape[0]
    
        # ratio positive/total
        len_positive_samples = train.shape[0] - len_negative_samples
        
        positive_ratio = len_positive_samples / number_of_samples
        
        # total number of row to remove
        samples_to_remove = number_of_samples - (len_positive_samples*2)
        indexes_to_remove = np.random.choice(negative_samples.index.values,samples_to_remove, replace=False)
        
        # we drop from the train dataset the index we want to remove
        # as a result of our under sampling
        train_set = train_samples.drop(indexes_to_remove)
        
        yield train_set, data.iloc[test]
    