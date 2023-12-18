import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

X_train, X_ivs, y_train, col_names = pickle.load(open("drd2_data.pickle", "rb"))
# Check size (number of features and samples)

print("Shape of X_train:", X_train.shape)

# Check for missing values
missing_val_Xtrain = np.isnan(X_train).sum()
missing_val_ytrain = np.isnan(y_train).sum()
missing_val_Xivs = np.isnan(X_ivs).sum()

print("Total missing values in X_train: ", missing_val_Xtrain)
print("Total missing values in y_train: ", missing_val_ytrain)
print("Total missing values in X_ivs: ", missing_val_Xivs)

# Split to new train and test, in order to be able to use the statistics
X_train_new, X_test, y_train_new, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=123)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test)
X_ivs_scaled = scaler.transform(X_ivs)

# Feature selection using Random Forest
N, M = X_train_scaled.shape

rfr = RandomForestRegressor(random_state=0)
selector = SelectFromModel(estimator=rfr, threshold="median")
selector.fit(X_train_scaled, y_train_new)

features = selector.get_support()
Features_selected = np.arange(M)[features]

print(f"There were {len(Features_selected)} features selected, from the original {M}")

# Applying the changes in the features that were selected
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
X_ivs_selected = selector.transform(X_ivs_scaled)

# Principal component analysis with all columns - to check how many PC to keep
n = len(Features_selected)
kpca = KernelPCA(n_components=n, kernel='poly')  # we will use poly since it keeps one less pc than rbf

kpca.fit(X_train_selected)

# Computing the variance explained
eigenvalues = kpca.eigenvalues_
explained_variance_ratio = eigenvalues / sum(eigenvalues)

# Print the variance for each PC as well as the total variance at that point
tve = 0

for i, ve in enumerate(explained_variance_ratio):
    tve += ve
    print("PC%d - Variance explained: %7.4f - Total Variance: %7.4f" % (i+1, ve, tve))

# Now we will apply the kernel pca with the number of PC we chose to keep

kpca = KernelPCA(n_components=236, kernel='poly')

kpca.fit(X_train_selected)
X_train_pca = kpca.transform(X_train_selected)
X_test_pca = kpca.transform(X_test_selected)
X_ivs_pca = kpca.transform(X_ivs_selected)


def present_statistics(y_test, preds, model):
    print(model, ':')
    print("The RVE is: %7.4f" % explained_variance_score(y_test, preds))
    print("The rmse is: %7.4f" % mean_squared_error(y_test, preds, squared=False))
    print("Mean Absolute Error: %7.4f" % mean_absolute_error(y_test, preds))
    print("Mean Squared Error: %7.4f" % mean_squared_error(y_test, preds))
    corr, pval = pearsonr(y_test, preds)
    print("The Correlation Score is: %6.4f (p-value=%e)" % (corr, pval))


# ---------------------------------------Linear regression----------------------------------------------
lnr = LinearRegression()
lnr.fit(X_train_pca, y_train_new)
preds = lnr.predict(X_test_pca)
present_statistics(y_test, preds, model='Linear Regression')

# -------------------------------------------Ridge------------------------------------------------------
alphas = [0.1, 0.001, 0.01, 0.0001, 0.00001, 1]
param_grid = {"alpha": alphas}
ridge = Ridge()
gs = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring="neg_mean_absolute_error")
gs = gs.fit(X_train_pca, y_train_new)
print("best alpha: %7.4f" % gs.best_estimator_.alpha)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Ridge Regression')

# ---------------------------------------------------Lasso-------------------------------------------------
alphas = [0.1, 0.001, 0.01, 1]
param_grid = {"alpha": alphas}
lasso = Lasso()
gs = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring="neg_mean_absolute_error")
gs = gs.fit(X_train_pca, y_train_new)
print("best alpha: %7.4f" % gs.best_estimator_.alpha)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Lasso Regression')

# -------------------------------- Decision Tree Regressor ------------------------------
max_depth = [3, 4, 5, 6, 7]
min_samples_leaf = [1, 2, 3, 4, 5, 10]
param_grid = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
dt = DecisionTreeRegressor(random_state=13)
gs = GridSearchCV(estimator=dt, param_grid=param_grid, scoring="neg_mean_absolute_error")
gs = gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Decision Tree Regressor')

# ------------------------ Support Vector Regression - kernel -------------------------

kernel = ['poly', 'rbf']
degree = [3, 4, 5, 6, 7]
epsilon = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
c = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.5]
param_grid = {'kernel': kernel, 'degree': degree, 'epsilon': epsilon, 'C': c}
svr = SVR()
gs = GridSearchCV(estimator=svr, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs = gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Support Vector Regression')

# ------------------------------ Linear Support Vector Regression -----------------------

epsilon = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
c = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.5]
loss = ['squared_epsilon_insensitive', 'epsilon_insensitive']
param_grid = {'loss': loss, 'epsilon': epsilon, 'C': c}
lin_svm = LinearSVR(dual='auto', max_iter=500000, random_state=13)
gs = GridSearchCV(estimator=lin_svm, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Linear Support Vector Regression')

# -------------------------------------- K Nearest Neighbours Regressor ------------------------------------


n = [3, 5, 6, 7, 8, 9, 10]
weights = ['uniform', 'distance']
param_grid = {'n_neighbors': n, 'weights': weights}
knn = KNeighborsRegressor()
gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs = gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='K Nearest Neighbours Regressor')

# -------------------------------------------- Random Forest Regressor ---------------------------------

min_samples_leaf = [1, 3, 5]
n_estimators = [50, 100, 120, 150]
max_depth = [4, 6, 8]
param_grid = {'min_samples_leaf': min_samples_leaf, 'n_estimators': n_estimators, 'max_depth': max_depth}
rfr = RandomForestRegressor(random_state=3)
gs = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs = gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.best_estimator_.predict(X_test_pca)
present_statistics(y_test, preds, model='Random Forest Regressor')

# ---------------------------------------- Ada Boost --------------------------------

estimator = [DecisionTreeRegressor(max_depth=3), LinearRegression(), KNeighborsRegressor(n_neighbors=3),
             LinearSVR(dual='auto')]
n_estimators = [50, 100, 150]
learning_rate = [0.01, 0.5, 1.0]
param_grid = {'estimator': estimator, 'n_estimators': n_estimators, 'learning_rate': learning_rate}
ada = AdaBoostRegressor(random_state=10)
gs = GridSearchCV(estimator=ada, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs.fit(X_train_pca, y_train_new)
print("The best parameters are:")
print(gs.best_params_)
preds = gs.predict(X_test_pca)
present_statistics(y_test, preds, model='Ada Boost')

# --------------------------------------- XGBoost ------------------------------------
booster = ['gbtree', 'dart', 'gblinear']
n_estimators = [10, 20, 50, 100, 150]
learning_rate = [0, 0.5, 1]
param_grid = {'booster': booster, 'n_estimators': n_estimators, 'learning_rate': learning_rate}
xgb = XGBRegressor(random_state=42)
gs = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
gs = gs.fit(X_train_pca, y_train_new)
preds = gs.best_estimator_.predict(X_test_pca)
print("The best parameters are:")
print(gs.best_params_)
present_statistics(y_test, preds, model='XGBoost')

# plotting the truth vs predicted plots, with the best hyperparameters found before, for the 3 best models
mdl = SVR(kernel='rbf', degree=3, epsilon=0.01, C=0.7)
mdl.fit(X_train_pca, y_train_new)
preds = mdl.predict(X_test_pca)
plt.scatter(y_test, preds, color='#C17CD9')
plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100))
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('Support Vector Regression')
plt.show()

mdl = KNeighborsRegressor(n_neighbors=3, weights='distance')
mdl.fit(X_train_pca, y_train_new)
preds = mdl.predict(X_test_pca)
plt.scatter(y_test, preds, color='#C17CD9')
plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100))
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('K Nearest Neighbors')
plt.show()

mdl = AdaBoostRegressor(estimator=KNeighborsRegressor(n_neighbors=3), n_estimators=50, learning_rate=0.5,
                        random_state=10)
mdl.fit(X_train_pca, y_train_new)
preds = mdl.predict(X_test_pca)
plt.scatter(y_test, preds, color='#C17CD9')
plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100))
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('AdaBoost')
plt.show()

# Creating the predictions file
svr = SVR(kernel='rbf', degree=3, C=0.7, epsilon=0.01)
svr.fit(X_train_pca, y_train_new)
preds = svr.predict(X_ivs_pca)
file1 = open('36.txt', 'w')
for i in preds:
    if i < 0:
        file1.write(str(0))
    else:
        file1.write(str(i))
    file1.write('\n')
file1.close()
