import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers


class OrdinalLogitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        penalty='l2', 
        *, 
        dual=False, 
        tol=0.0001, 
        C=1.0, 
        fit_intercept=True, 
        intercept_scaling=1, 
        class_weight=None, 
        random_state=None, 
        solver='lbfgs', 
        max_iter=100, 
        multi_class='auto', 
        verbose=0, 
        warm_start=False, 
        n_jobs=None, 
        l1_ratio=None
    ):
        self.penalty = penalty 
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept 
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = LogisticRegression(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)



class OrdinalRFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        n_estimators=100, 
        *, 
        criterion='gini', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        class_weight=None, 
        ccp_alpha=0.0, 
        max_samples=None
    ):
        self.n_estimators = n_estimators 
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf 
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = RandomForestClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)


class model1(K.Model):
    def __init__(self):
        super(model1, self).__init__()
        self.dense1 = layers.Dense(5, activation = 'softmax')
        
    def call(self, inputs):
        yhat = self.dense1(inputs)
        
        return yhat


class model3(K.Model):
    def __init__(self):
        super(model3, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(50, activation = 'tanh')
        self.dense3 = layers.Dense(5, activation = 'softmax')
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        yhat = self.dense3(h2)
        
        return yhat


class model5(K.Model):
    def __init__(self):
        super(model5, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(70)
        self.dense3 = layers.Dense(50, activation = 'tanh')
        self.dense4 = layers.Dense(25)
        self.dense5 = layers.Dense(5, activation = 'softmax')
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        h4 = self.dense4(h3)
        yhat = self.dense5(h4)
        
        return yhat


class ordinal(layers.Layer):
    def __init__(self, num_class):
        super(ordinal, self).__init__()
        self.num_class = num_class
        self.theta = tf.Variable(tf.cumsum(tf.random.uniform((1, num_class - 1)), axis = 1))
        self.dense = layers.Dense(1)
        
    def call(self, inputs):
        x = tf.expand_dims(self.theta, 0) - self.dense(inputs)
        cum_prob = tf.squeeze(tf.nn.sigmoid(x))
        prob = tf.concat([
            cum_prob[:, :1], 
            cum_prob[:, 1:] - cum_prob[:, :-1],
            1 - cum_prob[:, -1:]], axis = 1)
        
        return prob


class ord_model(K.Model):
    def __init__(self):
        super(ord_model, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(50, activation = 'tanh')
        # self.dense3 = layers.Dense(5, activation = 'relu')
        self.dense3 = ordinal(5)
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        # h3 = self.dense3(h2)
        yhat = self.dense3(h2)
        
        return yhat


class Logit(K.Model):
    def __init__(self):
        super(Logit, self).__init__()
        self.dense = layers.Dense(1)
    
    def call(self, inputs):
        p = 1 / (1 + tf.math.exp(-self.dense(inputs)))
        
        return p


class WeightedLogitLoss(K.losses.Loss):
    def __init__(self, alpha):
        super(WeightedLogitLoss, self).__init__()
        self.alpha = alpha
    
    def call(self, y_true, prob):
        y_pred = tf.math.log(prob / (1 - prob))
        
        cond = y_true == 1
        
        loss_ = tf.math.log(1 + tf.math.exp(- 2 * y_true * y_pred))
        loss = tf.where(cond, 2 * self.alpha * loss_, 2 * (1 - self.alpha) * loss_)
        
        return tf.reduce_mean(loss)


@tf.function
def ridge(weight, lambda_):
	penalty = tf.math.square(weight) * lambda_
	return  tf.reduce_sum(penalty)


class ridge_dense(K.layers.Layer):
    '''
        bias항 추가 필요
    '''
    def __init__(self, h, output_dim, lambda_, **kwargs):
        super(ridge_dense, self).__init__(**kwargs)
        self.input_dim = h.shape[-1]
        self.output_dim = output_dim
        self.lambda_ = lambda_
        self.ridge = ridge
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(self.input_dim, 1), 
                                                  dtype='float32'), 
                             trainable=True)
        
    def call(self, x):
        h = tf.matmul(x, self.w)
        self.add_loss(self.ridge(self.w, self.lambda_))
        
        return h
        

class RidgeLogit(K.Model):
    def __init__(self, h, output_dim, lambda_, **kwargs):
        super(RidgeLogit, self).__init__()
        self.dense = ridge_dense(h, output_dim, lambda_)
    
    def call(self, inputs):
        p = 1 / (1 + tf.math.exp(-self.dense(inputs)))
        
        return p