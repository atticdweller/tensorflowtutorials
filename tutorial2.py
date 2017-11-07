# Numpy is often used to load, manipulate and preprocess data.
import numpy as np
import tensorflow as tf 

#declare list of features. We only have one numeric feature
# other types of columbs that are more complicated and useful
feature_columns = [tf.feature_columns.numeric_column("x",shape=[1])]

# an estimator is the front end to invoke training (fitting) and evaluation
# (inference). there are many predefined types like linear regression,
# linear classification, and many neural netowrk classifiers and greressors.
# the following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
