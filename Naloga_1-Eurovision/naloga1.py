import csv
import numpy as np

REMOVED_ATTRIBUTES = ['Region', 'Song language', 'Artist', 'Song', 'English translation', 'Artist gender', 'Group/Solo', 'Place', 'Points', 'Host Country', 'Host region', 'Home/Away Country', 'Home/Away Region', 'Approximate Betting Prices']

def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    # Open the file and read its contents to a list
    file = open(file_name, "rt", encoding="latin1")
    table = []
    reader = csv.reader(file)
    for line in reader:
        table.append(line)

    # Clean the keys and create a dictionary
    table[0] = list(map(lambda x : str(x).strip(),table[0]))
    data_dict = dict.fromkeys(table[0])

    # Insert all values into the dictionary
    k = 0
    for key in data_dict.keys():
        for i in range(1,len(table)):
            data_dict[key] = [table[i][k]] if data_dict[key] == None else data_dict[key] + [table[i][k]]
        k += 1
    data_dict.pop('',None)

    # Remove attributes that are not needed
    for attr in REMOVED_ATTRIBUTES:
        data_dict.pop(attr,None)

    for key in data_dict.keys():
        print(data_dict[key])

    return data_dict




class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        pass

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        pass

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        pass

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """
        pass

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        pass


if __name__ == "__main__":
    DATA_FILE = "eurovision-final.csv"
    read_file(DATA_FILE)
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
