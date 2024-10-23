from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report
import matplotlib.pyplot as plt


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = KNeighborsClassifier()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)


print("The accuracy score is:" , accuracy_score(y_test,y_predict))
print("The classification report is:\n",classification_report(y_test,y_predict))



l1 =[]
k_values = (1,12)
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    l_predicted = model.predict(x_test)
    l1.append(accuracy_score(y_test,l_predicted))
print(l1)


plt.figure(figsize=(7,6))
plt.plot( k_values, l1, marker='*')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K value")
plt.show()

