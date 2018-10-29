import random as rnd
import numpy as np
from unidecode import unidecode
from collections import Counter
from numpy.linalg import norm
from itertools import combinations

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
        """ Generate k-mers for an input string """
        for i in range(len(input) - k + 1):
            yield input[i:i + k]


    def kmedoids(self,k = 5):
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
                    print("Switching, medoids now: ", keys_medoids)
                    print(sum_now)
                else:
                    tmp = non_medoids[j]
                    non_medoids[j] = keys_medoids[i]
                    keys_medoids[i] = tmp


    def find_closest_points(self,medoids):
        """ Finds closest points to the medoids"""

        for key in medoids.keys():
            medoids[key] = []
        #print(medoids)
        dist_mat = self.dist_mat.copy()
        lang_keys = list(self.languages.keys())
        medoid_keys = list(medoids.keys())

        # Select only columns that show distances from medoids to other countries
        medoids_idx = list(map(lambda k: list(self.languages.keys()).index(k),medoids.keys()))
        new_arr = self.dist_mat[:, medoids_idx]
        distances = np.zeros(len(medoid_keys))

        # Set the value for the same country pair to max, so its never selected
        for i in range(len(medoids_idx)):
            new_arr[medoids_idx[i]][i] = 1

        # Find the nearest neighbours and add them to the medoids
        for i in range(len(self.languages)):
            min_val = min(new_arr[i])
            min_idx = list(new_arr[i]).index(min_val)
            distances[min_idx] += min_val
            medoids[medoid_keys[min_idx]].append(lang_keys[i])

        return medoids,distances


    def get_dist_matrix(self,lang_freq):
        """ Create a distance matrix between all texts """
        keys = lang_freq.keys()
        for k1,i in zip(keys,range(len(keys))):
            for k2,j in zip(keys,range(len(keys))):
                self.dist_mat[i][j] = self.cosine_dist(lang_freq[k1],lang_freq[k2])

        return self.dist_mat



    def cosine_dist(self,a,b):
        """ Calculates the cosine distance between two dictionaries """

        # Get the kmers that are in both languags
        lang_union = a.keys() & b.keys()
        list_a = [a[key] for key in lang_union]
        list_b = [b[key] for key in lang_union]

        # FIXME je v redu norma??
        # Calculate cosine distance between relevant ones
        dist = 1 - np.dot(list_a,list_b) / (norm(list_a) * norm(list_b))
        #dist = 1 - np.dot(list_a,list_b) / (norm(list(a.values())) * norm(list(b.values())))
        return 0 if dist < 1e-9 else round(dist,6)




if __name__ == "__main__":
    # Read all the languages into a dictionary
    languages = {lang : unidecode(open("langs/" + lang + ".txt", "rt", encoding="utf8").read()).lower() for lang in LANGUAGES}
    kmedoids = KmedoidsClustering(languages)
    kmedoids.kmedoids()