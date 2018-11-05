import glob
import os
import random as rnd
import numpy as np
from unidecode import unidecode
from collections import Counter
from numpy.linalg import norm
import matplotlib.pyplot as plt
from heapq import nlargest

# Define languages to test
LANG_ROMAN = ["itn","frn","por","spn","rum"]
LANG_GERMAN = ["ger","dns","swd","dut","eng","nrn","ice"]
LANG_SLOVAN = ["slv","slo","src5","rus","blg","pql"]
LANG_OTHER = ["grk","lit","czc","mls","est","trk"]


LANGUAGES = LANG_ROMAN + LANG_GERMAN + LANG_SLOVAN + LANG_OTHER



class KmedoidsClustering:

    def __init__(self,lang_dict):
        self.languages = lang_dict
        self.dist_mat = np.zeros(shape=(len(self.languages), len(self.languages)))

    def kmers(self,input,k = 3):
        """
         Generate k-mers for an input string
        :param input: Text to create k-mers from
        :param k: Length of k-kmers
        :return:
        """
        for i in range(len(input) - k + 1):
            yield input[i:i + k]


    def kmedoids(self,k = 5):
        """
         Returns a list of clusters determined with the kmedoids method
        :param k: Number of medoids to initialize
        :return:
        """

        # Create a dictionary with language frequencies
        lang_freq = {lang : dict(Counter(self.kmers(self.languages[lang]))) for lang in self.languages}

        # Get initial k medoids
        medoids = dict.fromkeys(list(rnd.sample(lang_freq.keys(),k)))

        # Create a distance matrix, lang_freq keys are indices for rows and columns
        # The matrix is SYMMETRIC ACROSS THE DIAGONAL, WATCH OUT
        self.get_dist_matrix(lang_freq)

        # Initialize
        medoids,distances = self.find_closest_points(medoids)
        non_medoids = list(set(lang_freq.keys()) - set(medoids.keys()))
        keys_medoids = list(medoids.keys())
        sum_prev = sum(distances)
        sum_now = 0

        # Iterate through medoids and non medoids
        for i in range(len(medoids.keys())):
            for j in range(len(non_medoids)):
                tmp = keys_medoids[i]
                keys_medoids[i] = non_medoids[j]
                non_medoids[j] = tmp
                medoids = dict.fromkeys(keys_medoids)
                medoids, distances = self.find_closest_points(medoids)
                sum_now = sum(distances)

                # Swap if the sum of all distances is smaller -> better result
                if sum_now < sum_prev:
                    sum_prev = sum_now
                    best_medoid = medoids
                else:
                    tmp = non_medoids[j]
                    non_medoids[j] = keys_medoids[i]
                    keys_medoids[i] = tmp
        return [best_medoid[key] + [key] for key in best_medoid.keys()]


    def calc_silhuettes(self,clusters):
        """
         Calculate the silhuette for a cluster
        :param clusters: list of lists that represents clusters
        :return: dict with countries as keys and values as silhuettes
        """
        silhuettes = dict.fromkeys([item for sublist in clusters for item in sublist])
        lang_idx = list(self.languages.keys())
        dist_mat = self.dist_mat.copy()

        for cluster in clusters:
            if len(cluster) == 1:
                silhuettes[cluster[0]] = 0
                continue
            for el in cluster:
                cluster_without_el = cluster.copy()
                cluster_without_el.remove(el)
                el_idx = lang_idx.index(el)
                others_inside_idx = [lang_idx.index(key) for key in cluster_without_el]

                # Calculate the average distance from a country to the other countries in the same cluster
                avg_dist_inside = sum(dist_mat[el_idx,others_inside_idx]) / len(others_inside_idx)

                # Calculate the smallest average distance to all points in any other cluster (the next best fit cluster for the point)
                other_clusters = clusters.copy()
                other_clusters.remove(cluster)
                avg_outside_distances = []
                for oth_cluster in other_clusters:
                    oth_cluster_idxs = [lang_idx.index(key) for key in oth_cluster]
                    avg_outside_distances.append(sum(dist_mat[el_idx,oth_cluster_idxs]) / len(oth_cluster_idxs))

                avg_dist_outside = min(avg_outside_distances)
                silhuettes[el] = round((avg_dist_outside - avg_dist_inside) / max(avg_dist_outside,avg_dist_inside),4)
        return silhuettes




    def find_closest_points(self,medoids):
        """
         Finds closest points to the medoids
        :param medoids: Randomly initialized medoids
        :return: return the medoids with its clusters and the distances
        """

        for key in medoids.keys():
            medoids[key] = []
        dist_mat = self.dist_mat.copy()
        lang_keys = list(self.languages.keys())
        medoid_keys = list(medoids.keys())

        # Select only columns that show distances from medoids to other countries
        medoids_idx = list(map(lambda k: list(self.languages.keys()).index(k),medoids.keys()))
        new_arr = dist_mat[:, medoids_idx]
        distances = np.zeros(len(medoid_keys))

        # Set the value for the same country pair to max, so its never selected
        for i in range(len(medoids_idx)):
            new_arr[medoids_idx[i]][i] = 1

        # Set the values of other medoids to max, so there can't be a country in two clusters
        for i in range(len(medoids_idx)):
            for idx in medoids_idx:
                new_arr[idx][i] = 1

        # Find the nearest neighbours and add them to the medoids
        for i in range(len(self.languages)):
            min_val = min(new_arr[i])
            if min_val == 1: continue # Don't select any of the other medoids
            min_idx = list(new_arr[i]).index(min_val)
            distances[min_idx] += min_val
            medoids[medoid_keys[min_idx]].append(lang_keys[i])

        return medoids,distances


    def get_dist_matrix(self,lang_freq):
        """
         Create a distance matrix between all texts
        :param lang_freq: dictionary of kmers frequencies for a given language
        :return: distance matrix between all languages (symmetric over the diagonal)
        """
        keys = lang_freq.keys()
        for k1,i in zip(keys,range(len(keys))):
            for k2,j in zip(keys,range(len(keys))):
                self.dist_mat[i][j] = self.cosine_dist(lang_freq[k1],lang_freq[k2])

        return self.dist_mat



    def cosine_dist(self,a,b):
        """
        Calculates the cosine distance between two dictionaries
        :param a: dict of the first language
        :param b: dict of the second language
        :return: return the euclidean distance between the frequencies of kmers in languages
        """

        # Get the kmers that are in both languags
        lang_intersection = a.keys() & b.keys()
        list_a = [a[key] for key in lang_intersection]
        list_b = [b[key] for key in lang_intersection]

        # FIXME je v redu norma??
        # Calculate cosine distance between relevant ones
        dist = 1 - np.dot(list_a,list_b) / (norm(list_a) * norm(list_b))
        #dist = 1 - np.dot(list_a,list_b) / (norm(list(a.values())) * norm(list(b.values())))
        return 0 if dist < 1e-9 else round(dist,6)

    def kmedoids_avg(self,k = 5):
        """
        Calculates the silhuettes over 100 iterations with randomly initialized medoids to find the best and the worst one
        :param k: number of medoids
        :return: void
        """

        # Find the best and the worst clusters depending on the silhuette
        best_silh = dict()
        worst_silh = dict()
        best_clusters = []
        worst_clusters = []

        # The higher the silhuette the better the object matches it's cluster (ranges between -1 and 1)
        worst_avg_silh = 1
        best_avg_silh = -1

        iterations = 100

        for i in range(iterations):
            print("Starting kmedoits iteration ",i," of ",iterations,".")
            clusters = self.kmedoids(k=k)
            silhuettes = self.calc_silhuettes(clusters)
            avg_silh = sum(silhuettes.values()) / len(silhuettes.values())
            if avg_silh < worst_avg_silh:
                worst_avg_silh = avg_silh
                worst_silh = silhuettes
                worst_clusters = clusters
            if avg_silh > best_avg_silh:
                best_avg_silh = avg_silh
                best_silh = silhuettes
                best_clusters = clusters

        # Plot the best and the worst
        print("Finished finding best and worst silhuettes, worstAvg=",round(worst_avg_silh,3),"; bestAvg=",round(best_avg_silh,3))
        self.plot_silhuettes(worst_silh,worst_clusters,title = "Worst silhuettes")
        self.plot_silhuettes(best_silh,best_clusters, title = "Best silhuettes")

    def plot_silhuettes(self,silhuettes,clusters,title = "Silhuette"):
        """
        Plots the silhuettes
        :param silhuettes: dict with values as silhuettes and keys as labels
        :param clusters: clusters from which the silhuettes were calculates
        :param title: plot title
        :return: void
        """
        plt.close('all')
        for idx in range(len(clusters)):
            fig, ax = plt.subplots()
            y_pos = np.arange(len(clusters[idx]))

            items = sorted([(key,item) for key, item in silhuettes.items() if key in clusters[idx]], key=lambda tup: tup[1],reverse=True)

            values = [item for key,item in items]
            labels = [key for key,item in items]

            ax.barh(y=y_pos, width=values, color='blue')
            ax.set_yticks(y_pos)
            ax.set_xlabel("Silhuette")
            ax.set_title(title)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            plt.show()


    def determine_language(self,text,nbiggest = 3):
        """
        Return the probability of the given text being in a certain language (top k probabilities)
        :param text: given text to determine
        :param nbiggest:
        :return: The most probable languages with probabilities
        """

        # Create a dictionary with language frequencies
        lang_freq = {lang: dict(Counter(self.kmers(self.languages[lang]))) for lang in self.languages}


        # Get the kmers frequencies for the given text
        text_freq = dict(Counter(self.kmers(text)))

        # Calculate the similarity of the text to the texts in other languages (1-.. to show similarity
        distances = {lang : 1 - self.cosine_dist(text_freq,lang_freq[lang]) for lang in lang_freq.keys()}

        # Get the nbiggest most probable langues
        most_probable = nlargest(nbiggest,distances.items(),key= lambda tup: tup[1])

        return most_probable

    def determine_paragraph_lang(self):
        """
        Determines the languages of the paragraphs in folder /paragraph
        :return: void
        """
        for file_name in glob.glob("paragraphs/*"):
            name = os.path.splitext(os.path.basename(file_name))[0]
            text = unidecode(open(file_name,"rt",encoding="utf8").read().lower())
            print("Text in language: ",name)
            print("Detected languages with probability:")
            for lang,prob in self.determine_language(text):
                print(lang,":",prob)
            print()



if __name__ == "__main__":
    # Shuffle the LANGUAGES
    rnd.shuffle(LANGUAGES)

    # Read all the languages into a dictionary
    languages = {lang : unidecode(open("langs/" + lang + ".txt", "rt", encoding="utf8").read()).lower() for lang in LANGUAGES}
    kmedoids = KmedoidsClustering(languages)

    # Average kmedoids
    kmedoids.kmedoids_avg()

    # Determine languages
    kmedoids.determine_paragraph_lang()

