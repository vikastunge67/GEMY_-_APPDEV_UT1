import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD

#random data 
X = np.random.rand(10, 5)  #"""10 samples and 5 features"""

#SVD using TruncatedSVD from sklearn
svd = TruncatedSVD(n_components=2)
X_transformed = svd.fit_transform(X)

#Save the SVD model using pickle
with open('svd_model.pkl', 'wb') as file:
    pickle.dump(svd, file)

print("SVD model saved successfully!")

#Load the saved SVD model using pickle
with open('svd_model.pkl', 'rb') as file:
    loaded_svd = pickle.load(file)

#Use the loaded model to transform the data again
X_transformed_loaded = loaded_svd.transform(X)
print("Transformed data from the loaded model:")
print(X_transformed_loaded)
