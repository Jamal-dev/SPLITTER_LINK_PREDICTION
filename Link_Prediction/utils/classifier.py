import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()

df_2 = pd.read_csv('classic_link_prediction.csv')

#features    
X = df_2.iloc[:,1:] 
X_scaled = scaler.fit_transform(X)

#target
y = df_2['Connection']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 0, shuffle=True, test_size=0.5)


# scores = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']

# Initialize the classifiers
clf1 = SVC()
clf2 = KNeighborsClassifier()
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier(random_state=42)
clf5 = GradientBoostingClassifier()
clf6 = MLPClassifier(random_state=42,max_iter=1000)


# parameters for SVC
params1 = {}
params1['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
params1['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
params1['classifier__degree'] = [3,4,5]
params1['classifier'] = [clf1]

# parameters for KNeighborsClassifier
params2 = {}
params2['classifier__n_neighbors'] = [3,4,5,6,7,8,9,10,15,20,25,30]
params2['classifier__p'] = [1,2]
params2['classifier'] = [clf2]

# parameters for DecisionTreeClassifier
params3 = {}
params3['classifier__criterion'] = ['gini','entropy']
params3['classifier__max_depth'] = [5,10,15,25,30,None]
params3['classifier__min_samples_split'] = [2,5,7,10,15]
params3['classifier__max_features'] = [None,'auto','sqrt','log2']
params3['classifier'] = [clf3]

# parameters for RandomForestClassifier
params4 = {}
params4['classifier__n_estimators'] = [10, 20, 50, 100, 200, 300]
params4['classifier__criterion'] = ['gini','entropy']
params4['classifier__max_depth'] = [5,10,15,25,30,40,50,None]
params4['classifier__min_samples_split'] = [2,5,7,10,15]
params4['classifier__max_features'] = [None,'auto','sqrt','log2']
params4['classifier'] = [clf4]

# parameters for GradientBoostingClassifier
params5 = {}
params5['classifier__loss'] = ['deviance','exponential']
params5['classifier__learning_rate'] = [10**-3,10**-2, 10**-1]
params5['classifier__n_estimators'] = [10, 20, 50, 100, 200, 300]
params5['classifier__max_depth'] = [5,10,15,25,30,40,50,None]
params5['classifier__max_features'] = [None,'auto','sqrt','log2']
params5['classifier'] = [clf5]

# parameters for MLPClassifier
params6 = {}
params6['classifier__hidden_layer_sizes'] = [(100,),(5,5,),(10,10,),(5,5,5,),(10,10,10,),(10,15,10,),(15,15,15,)]
params6['classifier__activation'] = ['relu', 'tanh']
params6['classifier__learning_rate'] = ['constant', 'adaptive']
params6['classifier'] = [clf6]


# creating the pipline
pipeline = Pipeline([('classifier', clf1)])
params = [params1, params2, params3, params4, params5, params6]


# applying grid search
# gs = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring=scores, refit='roc_auc', verbose=3).fit(X_train, y_train)
gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc', verbose=3).fit(X_train, y_train)


# saving the grid search results for later use
joblib.dump(gs, 'grid_search_object.pkl')

print(gs.best_params_)


# writing the results to a csv and sorting the rows by f1 score
grid_results = joblib.load("grid_search_object.pkl")
results_df = pd.DataFrame(grid_results.cv_results_)
main_df = results_df[['params','mean_test_accuracy','rank_test_accuracy','mean_test_precision','rank_test_precision','mean_test_recall','rank_test_recall','mean_test_roc_auc','rank_test_roc_auc','mean_test_f1','rank_test_f1']]
main_df = main_df.sort_values(by=["rank_test_roc_auc"])


# extracting the classifier name from params column and set it as an index for more readability
classifiers = ['SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'MLPClassifier']

for index, row in main_df.iterrows():
    for i in classifiers:
        if str(row['params']).find(i) != -1:
            main_df.at[index,'classifier'] = i

main_df.set_index('classifier', inplace=True)

main_df.to_csv('grid_search_results')

# Plotting the results
plt.scatter(main_df["mean_test_roc_auc"],main_df["rank_test_roc_auc"], s = main_df["mean_test_f1_lime"])

plt.figure(figsize=(20,10))
plt.bar(main_df["classifier"],main_df["mean_test_roc_auc"])
plt.savefig("grid_search_result.png")