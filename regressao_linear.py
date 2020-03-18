import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, model_selection
from sklearn.metrics import mean_squared_error, r2_score

#x = np.random.random_sample(10,)
#y = np.random.random_sample(10,)
y = np.random.choice(1000,5000)
print('Valores: ', y)
bimonthly_days = np.arange(0, 1000)
base_date = np.datetime64('2016-01-01')

random_date = []
for x in range(len(y)):
    random_date.append(base_date + np.random.choice(bimonthly_days))  
#[print(x) for x in random_date]
x = np.asarray(random_date).astype(int)

df  = pd.DataFrame()
df['data'] = x
df['venda'] = y
print(df.shape)
#print(df.head(5))

train_X, test_X,train_Y,test_Y = model_selection.train_test_split(df[['data']], df[['venda']], test_size=0.3)
#Train model
model = linear_model.LinearRegression()
model.fit(train_X, train_Y)

#split to test
# test_X, test_Y = df[['data']][70:], df[['venda']][70:]
predcit_y = model.predict(test_X)

print('{0:0.2f}% nos dados de treino'.format((len(train_X)/len(df.index))*100))
print('{0:0.2f}% nos dados de teste'.format((len(test_X)/len(df.index))*100))

print('Retorno Treino')
print('Coeficiente: %.2f'% model.coef_)
print('Erro quad: %.2f' %mean_squared_error(test_Y,predcit_y))
print('Coef det: %.2f' %r2_score(test_Y, predcit_y))
print()
plt.scatter(test_X, test_Y,  color='red')
plt.plot(test_X, predcit_y, color='blue', linewidth=3)
plt.xticks(())
plt.yticks()
plt.ylabel('valor venda')

plt.show()











# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]
# #print(diabetes_X)


# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# #Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: ', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()