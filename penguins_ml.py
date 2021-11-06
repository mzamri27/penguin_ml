import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g','sex']]

# one hot encoding for feature
features = pd.get_dummies(features)
# encoding categorical data in output df
output, uniques = pd.factorize(output)
# train test split
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size= .2)
# classifier setup
rfc = RandomForestClassifier(random_state=15)
# modeling the x_train and y_train
rfc.fit(x_train, y_train)
# get the prediction
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))

# save the Penguin classifier RF model into pickle file
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()
