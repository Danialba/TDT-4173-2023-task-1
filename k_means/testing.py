import numpy as np
import pandas as pd


df=pd.read_csv("/Users/danialbashir/Desktop/Projects/TDT-4173-2023-task-1/k_means/data_1.csv")
df2=pd.read_csv("/Users/danialbashir/Desktop/Projects/TDT-4173-2023-task-1/k_means/data_1_copy.csv")


def fit(X):

    centroids=pd.DataFrame()
    new_centroids=X.sample(3)
    
    cluster_a=[]
    cluster_b=[]
    cluster_c=[]
    
    while not centroids.equals(new_centroids):
        for n in range (len(X)):
            print(n)
            sample= X.iloc[n].to_numpy()
            a= euclidean_distance(new_centroids.iloc[0].to_numpy(), sample)
            b= euclidean_distance(new_centroids.iloc[1].to_numpy(), sample)
            c= euclidean_distance(new_centroids.iloc[2].to_numpy(), sample)
            
            if min(a,b,c)==a:
                cluster_a.append(sample)
            elif min(a,b,c)==b:
                cluster_b.append(sample)
            elif min(a,b,c)==c:
                cluster_c.append(sample)
       
        centroids=new_centroids        
        new_centroids = pd.DataFrame([mean(cluster_a), mean(cluster_b), mean(cluster_c)])


def euclidean_distance(x, y):

    return np.linalg.norm(x - y, ord=2, axis=-1)

def mean(dataset):
    
            # Convert the list of lists to a NumPy array
    dataset_array = np.array(dataset)

    # Calculate the mean along each dimension (axis)
    return np.mean(dataset_array, axis=0)
    

fit(df2)

print(df2)
print(df2.to_numpy().tolist())