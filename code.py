import math
from matplotlib import pyplot as plt
import numpy as np


def parse_file_to_objects(filepath):
    objects = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue 
            try:
                x = float(parts[0])
                y = float(parts[1])
                objects.append({'x': x, 'y': y})
            except ValueError:
                continue 
    return objects


def plot_unclustered_data(g1,g2,g3):
    g1_x = [point['x'] for point in g1]
    g1_y = [point['y'] for point in g1]

    g2_x = [point['x'] for point in g2]
    g2_y = [point['y'] for point in g2]

    g3_x = [point['x'] for point in g3]
    g3_y = [point['y'] for point in g3]

    plt.figure(figsize=(10, 6))
    plt.scatter(g1_x, g1_y, label="Group 1", alpha=0.6)
    plt.scatter(g2_x, g2_y, label="Group 2", alpha=0.6)
    plt.scatter(g3_x, g3_y, label="Group 3", alpha=0.6)
    plt.title("Parsed Data from Files")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_original_vs_clusters(g1, g2, g3, clusters):
    g1_x = [point['x'] for point in g1]
    g1_y = [point['y'] for point in g1]
    g2_x = [point['x'] for point in g2]
    g2_y = [point['y'] for point in g2]
    g3_x = [point['x'] for point in g3]
    g3_y = [point['y'] for point in g3]

    colors = [
        (0.1216, 0.4667, 0.7059, 1.0), (1.0, 0.498, 0.0549, 1.0), (0.1725, 0.6275, 0.1725, 1.0), (0.8392, 0.1529, 0.1569, 1.0),
        (0.5804, 0.4039, 0.7412, 1.0), (0.549, 0.3373, 0.2941, 1.0), (0.8902, 0.4667, 0.7608, 1.0), (0.498, 0.498, 0.498, 1.0),
        (0.7373, 0.7412, 0.1333, 1.0), (0.0902, 0.7451, 0.8118, 1.0), (0.6824, 0.7804, 0.9098, 1.0), (1.0, 0.7333, 0.4706, 1.0),
        (0.5961, 0.8745, 0.5412, 1.0), (1.0, 0.5961, 0.5882, 1.0), (0.7725, 0.6902, 0.8353, 1.0), (0.7686, 0.6118, 0.5804, 1.0),
        (0.9686, 0.7137, 0.8235, 1.0), (0.7804, 0.7804, 0.7804, 1.0), (0.902, 0.902, 0.498, 1.0), (0.5019, 0.8509, 0.8784, 1.0),
        (0.1922, 0.2118, 0.5843, 1.0), (0.2706, 0.4588, 0.7059, 1.0), (0.4549, 0.6784, 0.8196, 1.0), (0.6196, 0.7922, 0.8824, 1.0),
        (0.7765, 0.8588, 0.9373, 1.0), (0.902, 0.9294, 0.9686, 1.0), (0.1725, 0.2392, 0.1725, 1.0), (0.2549, 0.4275, 0.2549, 1.0),
        (0.4353, 0.6627, 0.4353, 1.0), (0.5961, 0.7686, 0.5961, 1.0), (0.7765, 0.8588, 0.7765, 1.0), (0.902, 0.9294, 0.902, 1.0),
        (0.349, 0.2078, 0.2314, 1.0), (0.549, 0.4275, 0.451, 1.0), (0.7686, 0.651, 0.6667, 1.0), (0.8588, 0.7765, 0.7843, 1.0),
        (0.9373, 0.902, 0.9059, 1.0), (0.9882, 0.9647, 0.9647, 1.0), (0.2863, 0.2902, 0.1961, 1.0), (0.4549, 0.4588, 0.2863, 1.0)
    ]

    clustered_points = [np.array(c) for c in clusters]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(g1_x, g1_y, label="Group 1", color=colors[2 % 10], alpha=0.6)
    axs[0].scatter(g2_x, g2_y, label="Group 2", color=colors[1 % 10], alpha=0.6)
    axs[0].scatter(g3_x, g3_y, label="Group 3", color=colors[0 % 10], alpha=0.6)
    axs[0].set_title("Original Groups")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()
    axs[0].grid(True)

    for i, cluster in enumerate(clustered_points):
        axs[1].scatter(cluster[:, 0], cluster[:, 1],
                       color=colors[i], alpha=0.6, label=f"Cluster {i+1}")
    axs[1].set_title("Clusters from Divisive Clustering")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    #axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()



def dissimilarity(point1, point2):
    return math.dist(point1, point2)

def average_dissimilarity(point,data):
 
    average = 0

    for i in range(0, len(data)):

        average += dissimilarity(point,data[i])

    return average / len(data)



def divisive_clustering(cluster):
    #Step 2: Select the data point with the greatest average disimilarity and place it as the first instance in a new cluster
    avg_data = list(map(lambda x: average_dissimilarity(x, cluster), cluster)) #average_dissimilarities(cluster,cluster)
    idx = np.argmax(avg_data)

    new_cluster = [cluster[idx]]
    cluster = np.delete(cluster, idx, axis=0)
    updated_cluster = []

    transfers = 0
    #Step 3: Reassign elements from one cluster to the other,
    #       as long as their average dissimilarity to the first 
    #       is greater that their distance to the new one
    for i in range(0,len(cluster)):
        avg_new_cluster = average_dissimilarity(cluster[i],new_cluster)
        avg_updated_cluster = average_dissimilarity(cluster[i],cluster)
        if avg_updated_cluster > avg_new_cluster:
            new_cluster.append(cluster[i])
            transfers += 1
        else: 
            updated_cluster.append(cluster[i])

    return [updated_cluster, new_cluster, transfers]



def cluster_data(data):
    #Step 1: Place all data in one cluster
    c0 = data
    clusters = [c0]

    #Step 4: Repeat the same process until all cluster have only one data point,
    #       or until no more transfers are possible.
    transfers = -1
    while not all(len(cluster) == 1 for cluster in clusters) and not transfers==0:
        largest_cluster_index = np.argmax([len(c) for c in clusters])
        updated_cluster, new_cluster, t = divisive_clustering(clusters[largest_cluster_index])

        clusters[largest_cluster_index] = updated_cluster
        clusters.append(new_cluster)
        transfers = t

    return clusters


group1 =  parse_file_to_objects("om1.txt")
group2 =  parse_file_to_objects("om2.txt")
group3 =  parse_file_to_objects("om3.txt")
all_points = group1 + group2 + group3
data = np.array([[p['x'], p['y']] for p in all_points])

clusters = cluster_data(data)
plot_original_vs_clusters(group1,group2,group3,clusters)
