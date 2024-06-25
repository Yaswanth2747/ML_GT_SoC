import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def VAR(l1):
    n = len(l1)
    variance = sum((x-np.mean(l1)) ** 2 for x in l1) / n
    return variance

def COV(l1, l2):
    xm = np.mean(l1)
    ym = np.mean(l2)
    sum = 0
    n = len(l1)
    for i in range(n):
        sum += (l1[i]-xm)*(l2[i]-ym)
    sum = sum/n
    return sum

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    #print(init_array)

    feature_data=[]
    
    for i in range(4):
        col = init_array[i].tolist()
        col = [x - np.mean(col) for x in col]
        feature_data.append(col)
    
    feature_arr=pd.DataFrame(feature_data)
    feature_arr=feature_arr.T
    #feature_arr.to_csv('feature_arr.csv',header=False,index=False)
    #print(feature_arr)

    cov_mat = pd.DataFrame(index=range(4), columns=['f1', 'f2', 'f3', 'f4'])
    for x in range(4):
        for y in range(4):
            if x == y:
                cov_mat.iloc[x , y] = VAR(feature_arr[y].tolist())
            else :
                cov_mat.iloc[x , y] = COV(feature_arr[x].tolist(),feature_arr[y].tolist())

    #print(cov_mat)
    eigen_mat = cov_mat.to_numpy().astype(float)
    #print(eigen_mat)
    eigenvalues, eigenvectors = np.linalg.eig(eigen_mat)
    #print(eigenvalues)
    #print(eigenvectors)
    sorted_eigenvalues=eigenvalues.tolist()
    sorted_eigenvalues.sort(reverse=True)
    sorted_eigenvalues=[round(x,4) for x in sorted_eigenvalues]
    #print(sorted_eigenvalues)
    eig_top=[]
    eig_top.append(eigenvectors[0].tolist())
    eig_top.append(eigenvectors[1].tolist())
    eig_mat=pd.DataFrame(eig_top)
    eig_mat=eig_mat.T
    #print(eig_mat)
    # END TODO
    rows, columns = eig_mat.shape
    #print("Number of rows:", rows)
    #print("Number of columns:", columns)
    final_data=feature_arr.dot(eig_mat)
    print(final_data)
    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png

first_column = final_data[1].tolist()  
second_column = final_data[0].tolist()
  
plt.scatter(first_column,second_column)
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.savefig("scatter.png")
plt.show()
    # END TODO
