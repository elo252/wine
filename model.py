# #Data
# import pandas as pd

# dataset = pd.read_csv(r'data.csv') #Any kind of data is accepted
# dataset.head()



# #Model Interpretation
# import lime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
# #from sklearn.linear_model import LogisticRegression
# #from sklearn.linear_model import LinearRegression
# import warnings
# warnings.filterwarnings('ignore')
# import math
# %matplotlib inline
# from lime import lime_tabular

# import pickle



# category = pd.cut(dataset.quality,bins=[0,5,10,],labels=['Bad','Good'])
# dataset.insert(12,'Result',category)
# #dataset['Result'].value_counts(normalize=True)


# dataset.head()


# dataset.drop('quality', axis = 1, inplace= True)

# #Train/Test Data
# #Training
# from sklearn.ensemble import RandomForestClassifier


# model = RandomForestClassifier(random_state= 50)
# model.fit(X_train, y_train)
# score = model.score(X_test, y_test)



# pickle.dump(model,open('model.pkl', 'wb'))
# model_2 = pickle.load(open('model.pkl','rb'))

# #Training
# from sklearn.ensemble import RandomForestClassifier


# model = RandomForestClassifier(random_state= 50)
# model.fit(X_train, y_train)
# score = model.score(X_test, y_test)



# pickle.dump(model,open('model.pkl', 'wb'))
# model_2 = pickle.load(open('model.pkl','rb'))


# dataset.head(25)



# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=np.array(X_train),
#     feature_names=X_train.columns,
#     class_names=['Bad', 'Good'],
#     mode='classification'
# )



# exp = explainer.explain_instance(
#     data_row=X_test.iloc[18],
#     predict_fn=model.predict_proba
# )

# exp.show_in_notebook(show_table=True)



# exp = explainer.explain_instance(
#     data_row=X_test.iloc[4],
#     predict_fn=model.predict_proba
# )

# exp.show_in_notebook(show_table=True)


# dataset.isnull().sum()

# dataset['Result'].value_counts()


