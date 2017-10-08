from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from logger import logger

# Set the parameters by cross-validation



def evaluate_estimator(X_train, X_test, y_train, y_test, cl, parameters):
    scores = ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
              'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error',
              'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2',
              'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

    for score in scores:
        logger.info("# Tuning hyper-parameters for %s" % score)
        logger.info('START:********************%s********************'% score)
        try:
            clf = GridSearchCV(cl, parameters, cv=10, scoring=score)
            clf.fit(X_train, y_train)
            logger.info("Best parameters set found on development set:")
            logger.info(clf.best_params_)
            # logger.info("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     logger.info("%0.3f (+/-%0.03f) for %r"
            #                 % (mean, std * 2, params))

            logger.info("Detailed classification report:")

            logger.info("The model is trained on the full development set.")
            logger.info("The scores are computed on the full evaluation set.")

            y_true, y_pred = y_test, clf.predict(X_test)
            logger.info(classification_report(y_true, y_pred))
        except Exception as e:
            logger.info(e)
        logger.info('END:********************%s********************\n'% score)


def do():
    # Loading the Digits dataset
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    cl = LinearSVC(C=1)
    evaluate_estimator(X_train, X_test, y_train, y_test, cl, lsvc_tuned_parameters)


if __name__ == '__main__':
    do()
