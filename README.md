lily
====

# A small ML Library

Includes

* k-means
* naive Bayes
* decision trees
* logistic regression
* support vector machines

## Examples

### k-means

```
from lily import k_means

def load_dataset(filepath):
    dataset = []
    fr = open(filepath)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        float_line = map(float, current_line)
        dataset.append(float_line)
    return dataset

if __name__ == '__main__':
	data_matrix = np.mat(load_dataset('data/test_data.tsv'))

    #random centroid
    rand_cent = k_means.random_centroid(data_matrix, 2)

    #euclidean distance
    euc_dist = k_means.euclidean_distance(data_matrix[0], data_matrix[1])

    #vanilla k-means
    centroids, cluster_assignment = k_means.k_means(data_matrix, 4)

    #bisecting k-means
    centroid_list, assessments = k_means.bisect_k_means(data_matrix, 3)
```

## Named after Lily

<img src="http://media.giphy.com/media/21KuU51onRVHG/giphy.gif" alt="lily"/>
