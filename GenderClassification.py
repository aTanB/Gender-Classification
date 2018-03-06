

from sklearn import tree, neighbors


#Dataset
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#Initializing Decision Tree
clf = tree.DecisionTreeClassifier()
#Training the decision Tree
clf = clf.fit(X,Y)

prediction = clf.predict([[100,70,43]])

print(prediction)


##################################################################################
#KNeighborsClassifier


h = 0.02 #Step size

clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X,Y)

Z = clf.predict([[100,70,43]])
print(Z)


