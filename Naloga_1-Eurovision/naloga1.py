import csv
import math
import sys

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

    # Average the missing attributes from the votes to the country of previous years
    cnt = 0
    for key in data_dict.keys():
        idx = 0
        for el in data_dict[key]:
            if not el and data_dict["Country"][idx] != key:
                country_row = data_dict["Country"][idx]
                curr_avg = calculate_average(data_dict,key,country_row)
                for i in range(idx,len(data_dict[key])):
                    if(data_dict["Country"][i] == country_row):
                        data_dict[key][i] = curr_avg
                        cnt += 1
            idx += 1
    # 4612/13677 -> 34% missing
    # Remove the keys we don't need anymore
    data_dict.pop("Year",None)
    data_dict.pop("Country",None)
    return data_dict

def calculate_average(data_dict,key,country):
    """ Calculate the average for a country-country pair score to fill in the missing votes """
    numbers = []
    for i in range(len(data_dict[key])):
        if data_dict["Country"][i] == country and data_dict[key][i]:
            numbers.append(int(data_dict[key][i]))
    return round(sum(numbers)/len(numbers)) if len(numbers) != 0 else 0



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
        _sum = 0
        for i in range(len(self.data[r1])):
            if self.data[r1][i] and self.data[r2][i]:
                _sum += (int(self.data[r1][i]) - int(self.data[r2][i]))**2

        return math.sqrt(_sum)

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        c1_items = self.get_nested_items(c1)
        c2_items = self.get_nested_items(c2)
        _sum = 0
        for el1 in c1_items:
            for el2 in c2_items:
                _sum += self.row_distance(el1,el2)
        return _sum/(len(c1_items) * len(c2_items))

    def get_nested_items(self, lst):
        """ Retrieve all the items that in a list with nested lists """
        items = []
        for el in lst:
            if type(el) == type([]):
                items = self.get_nested_items(el) + items
            else:
                items.append(el)
        return items



    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        _min = sys.maxsize
        c1,c2 = [],[]
        for i in range(len(self.clusters)):
            for j in range(i, len(self.clusters)):
                dist = self.cluster_distance(self.clusters[i],self.clusters[j])
                if(dist < _min and self.clusters[i] != self.clusters[j]):
                    _min = dist
                    c1 = self.clusters[i]
                    c2 = self.clusters[j]
        return c1,c2,_min



    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """
        distances = []
        while len(self.clusters) > 2:
            closest1,closest2,min_dist = self.closest_clusters()
            # Add a tuple for the distances
            distances.append((closest1,closest2,round(min_dist,3)))
            idx1 = self.clusters.index(closest1)
            idx2 = self.clusters.index(closest2)
            # Delete those clusters that have been merged and put them in the back (watch out for indices)
            del self.clusters[idx1],self.clusters[idx2 if idx1 > idx2 else idx2-1]
            self.clusters.append([closest1,closest2])




    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        self.plot_tree_rec(self.clusters,0)

    def plot_tree_rec(self,lst,depth):
        """ Recursive function to draw the tree """
        if(len(lst) == 1):
            print("    "*depth,"----",lst[0])
        else:
            self.plot_tree_rec(lst[0],depth+1)
            print("    "*depth,"----|")
            self.plot_tree_rec(lst[1],depth+1)



if __name__ == "__main__":
    DATA_FILE = "eurovision-final.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
