import pandas as pd
from CTL.CTL import CausalTree
import numpy as np
from sklearn.model_selection import train_test_split

asthma = pd.read_csv('data/asthma.txt', delimiter=' ', index_col=None)

asthma.columns = ['physician', 'age', 'sex', 'education', 'insurance', 'drug coverage', 'severity',
                  'comorbidity', 'physical comorbidity', 'mental comorbidity', 'satisfaction']

y = asthma['satisfaction'].values
treatment = asthma['physician'].values

x = asthma.drop(['satisfaction', 'physician'], axis=1).values

columns = asthma.drop(['satisfaction', 'physician'], axis=1).columns

y[y == 0] = -1

treatment[treatment == 1] = 0
treatment[treatment == 2] = 1

np.random.seed(10000)

treatment = np.random.randn(y.shape[0])

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

ct = CausalTree(cont=True)
ct.fit(x_train, y_train, treat_train)
effect_prediction = ct.predict(x_test)
# print(effect_prediction)
