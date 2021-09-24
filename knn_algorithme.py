# -*- coding: utf-8 -*-
#%% On importe les bibliothèques
import warnings
warnings.filterwarnings("ignore")

import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.exceptions import NotFittedError
from scipy.stats import mode

#%% On définit la méthode de normalisation
standard = StandardScaler()
#%% On importe les datasets d'entrainements
colnames = [f"Var{i}" for i in range(1,7)]
colnames.append("Class")
dataset = pd.read_csv("data.csv",names=colnames)
dataset["Class"] = dataset["Class"].apply(lambda x: x[-1])
dataset.head()

colnames = [f"Var{i}" for i in range(1,7)]
colnames.append("Class")
dataset_pretest = pd.read_csv("preTest.csv",names=colnames)
dataset_pretest["Class"] = dataset_pretest["Class"].apply(lambda x: x[-1])
dataset_pretest.head()

#%% On définit la distance euclidienne
def euclidian_distance_(p1,p2):
  return sum([(i-j)**2  for i, j in zip(p1,p2)])**0.5
  
def convert2numpy(data):
  if isinstance(data, pd.core.frame.DataFrame):
    return data.to_numpy()
  elif isinstance(data, list): 
    return np.array(data)
  elif isinstance(data,np.ndarray):
    return data
  else:
    raise TypeError("Les données doivent être des types suivants:\npandas.core.frame.DataFrame\nlist\nnumpy.ndarray")
 
class DataError(Exception):
  """
  Les labels sont présents dans les données
  """
  def __init__(self, name):
    self.name = name

  def __str__ (self):
      return f"Les labels sont présents dans les données."

class knn_:
  def __init__(self, k, metric = euclidian_distance_):
    self.data = None
    self.k = k if k > 0 else 1  
    self.metric = metric
    
  # Cette méthode permet de fournir les données d'entrainement au modèle
  def fit(self, train):
    self.data = convert2numpy(train)
    if isinstance(train, pd.core.frame.DataFrame):
      self.target_names=sorted(train.iloc[:,-1].unique())
    elif isinstance(train, (np.ndarray,list)):
      self.target_names=sorted(pd.DataFrame(train).iloc[:,-1].unique())
  
  
  def set_target_names(self, liste):
    if isinstance(liste, (list,np.ndarray)):
      self.target_names_ = liste
  def get_target_names(self):
    return self.target_names_
  # Propriété représentant la liste des classes de la variable cible
  target_names = property(get_target_names, set_target_names)

  # Méthode pour évaluer le modèle
  def score(self, test):
    if self.data is not None:
      test = convert2numpy(test)
      bon_val = 0
      for x in test:
        k_neighbors = sorted(self.data,key = lambda xi: self.metric(xi[:-1],x[:-1]))[:self.k]
        bon_val += 1 if mode(k_neighbors).mode[0][-1] == x[-1] else 0
      return bon_val/len(test)
    else:
      raise NotFittedError(f"This knn_ instance is not fitted yet. Call 'fit' with appropriate arguments before using the '{self.score.__name__}' function.")
      
  # Méthode pour la Validation croisée
  def crossValidation(self,data, fold_nb = 5, test_size = 0.25, random_state=0):
    cv = ShuffleSplit(n_splits=fold_nb, test_size=test_size, random_state=random_state)
    cv = cv.split(data)
    resultat = list()
    for train,test in cv:
      train = data.iloc[train,:]
      test = data.iloc[test,:]
      self.fit(train)
      resultat.append(self.score(test))
    return resultat

  # Méthode pour trouver le k optimal
  def find_best_k(self, validation, k_min = 1, k_max = 15):
    if self.data is not None:
      if isinstance(k_min, int) and isinstance(k_max, int):
        if k_min > 0 and k_max > k_min:
          best_params_values = dict()
          for k in range(k_min, k_max):
            self.k = k
            best_params_values[k] = self.score(validation)
          best = max(best_params_values.items(),key=lambda x: x[1])
          self.k = best[0]
          return best
        else:
          raise ValueError("k_min ou k_max n'est pas positif")
      else:
        raise TypeError("k_min ou k_max n'est pas un entier")
    else:
      raise NotFittedError(f"This knn_ instance is not fitted yet. Call 'fit' with appropriate arguments before using the '{self.score.__name__}' function.")

  # Méthode pour prédire des données inconnues
  def predict(self, X, save = True):
    if self.data is not None:
      X = convert2numpy(X)
      if type(X[0][-1]) is not str:
        if not isinstance(X[0],np.ndarray):
          k_neighbors = sorted(self.data,key = lambda xi: self.metric(x,X))[:self.k]
          return mode(k_neighbors).mode[0][-1]
        else:
          val = list()
          for x in X:
            k_neighbors = sorted(self.data,key = lambda xi: self.metric(xi,x))[:self.k]
            val.append(mode(k_neighbors).mode[0][-1])

          if save is not None:
            with open(f"deseure--charron_sample.txt", "w") as f:
              f.writelines(list(map(lambda element: f"class{element}\n",val[:-1])))
              f.write(f"class{val[-1]}")
          return val
      else:
        raise DataError("Error")
    else:
      raise NotFittedError(f"This knn_ instance is not fitted yet. Call 'fit' with appropriate arguments before using the '{self.predict.__name__}' function.")

  def convert_labels_to_int(self,y):
    return self.target_names.index(y)

  # Méthode pour la matrice de confusion
  def confusion_matrix_(self,y_true, y_pred):
    mat = np.zeros((len(self.target_names),len(self.target_names)))
    for i,j in zip(y_true, y_pred):
      mat[self.convert_labels_to_int(i[-1])][self.convert_labels_to_int(j)] += 1
    return mat

# Méthode fourni pour la vérification du fichier de rendu
def isCorrect(testset):
    allLabels = ['classA','classB','classC','classD','classE']
    nbLines = len(testset)
    with open('deseure--charron_sample.txt','r') as fd:
        lines = fd.readlines()
    count=0
    for label in lines:
        if label.strip() in allLabels:
            count+=1
        else:
            if count<nbLines:
                print("Wrong label line:"+str(count+1))
            break
    if count<nbLines:
        print(count,nbLines)
        return False
    else:
        return True

#%% On évalue les données 
# Création du dataset
data = pd.concat([dataset,dataset_pretest],axis=0).reset_index(drop=True)
df = pd.DataFrame(standard.fit_transform(data.iloc[:,:-1].drop("Var5", axis=1)))
df.columns = data.columns.tolist()[:4] + ["Var6"]
df["Class"] = data.iloc[:,-1]

# Split des données
trainset, testset = train_test_split(df,test_size = 0.25, random_state = 0)

# Définition du modèle et évaluation
model_knn = knn_(5)
model_knn.fit(trainset)
print("Validation non croisée (données de test): ",round(model_knn.score(testset)*100,2), "%",sep="")
resultat = model_knn.crossValidation(df)
print("Validation croisée:",
      f"Moyenne = {round(np.mean(resultat)*100,2)}%",
      f"Ecart type = {round(np.std(resultat)*100,2)}%",
      f"Intervalle = [{round(min(resultat)*100,2)}, {round(max(resultat)*100,2)}]",sep="\n")

#%% Méthode pour la prédiction
def prediction(filepath,save = True):
  # Création du dataset de test
  colnames = [f"Var{i}" for i in range(1,7)]
  dataset_final = pd.read_csv(filepath,names=colnames)
  dataset_final.head()
  data = pd.concat([dataset,dataset_pretest],axis=0).reset_index(drop=True)
  
  # Normalisation du dataset
  df = pd.DataFrame(standard.fit_transform(data.iloc[:,:-1].drop("Var5", axis=1)))
  df.columns = data.columns.tolist()[:4] + ["Var6"]
  df["Class"] = data.iloc[:,-1]
  
  # Split des données
  trainset, _ = train_test_split(df ,test_size = 0.25, random_state = 0)

  df_final = dataset_final.drop("Var5", axis=1)
  df_final = pd.DataFrame(standard.fit_transform(df_final))

  # On initialise le modèle et on prédit les valeurs
  best_k = 5
  model = knn_(best_k)
  model.fit(trainset)
  result = model.predict(df_final,True)
  if isCorrect(df_final):
      print("\nPrédiction et enregistrement:")
      print("Labels Check : Successfull!")
      result_dico = {classe:nb for classe, nb in zip(np.unique(result, return_counts=True)[0],np.unique(result, return_counts=True)[1])}
      result_dico = sorted(result_dico.items(), key=lambda x: x[1], reverse = True)
      for classe, nb in result_dico:
          print(classe, nb, sep = "    ") 
      print("\nProportion df_final")
      for classe, nb in result_dico:
          print(classe, round(nb/len(df_final),6), sep = "    ") 
      print("\nProportion data")
      print(dataset["Class"].value_counts(normalize = True))
      print("\nProportion pre_test")
      print(dataset_pretest["Class"].value_counts(normalize = True))
      print("\nOn peut calculer la différence de proportion entre les données de preTest et les données prédites:", end=" ")
      result_nb = [nb[1]/len(df_final) for nb in result_dico]
      erreurs = list()
      for originale, predite in zip(dataset_pretest["Class"].value_counts(normalize = True),result_nb):
          erreurs.append(abs(predite-originale)) 
      print("%.2f"%(sum(erreurs)*100),"% d'erreur.")
  else:
      print("Labels Check : fail!")
#%% On effectue la prédiction
prediction("finalTest.csv")