from unidecode import unidecode

# Define languages to test
LANG_ROMAN = ["itn","frn","por","spn","rum"]
LANG_GERMAN = ["ger","dns","swd","dut","eng","nrn","ice"]
LANG_SLOVAN = ["slv","slo","src5","src4","rus","blg","pql"]
LANG_OTHER = ["grk","lit","czc","mls","est","trk"]


LANGUAGES = LANG_ROMAN + LANG_GERMAN + LANG_SLOVAN + LANG_OTHER



class KmedoidsClustering:

    def __init__(self,lang_dict):
        self.languages = lang_dict

    def kmedoids(self,k = 3):
        pass

    def cosine_dist(self):
        pass


if __name__ == "__main__":
    # Read all the languages into a dictionary
    languages = {lang: unidecode(open("langs/" + lang + ".txt", "rt", encoding="utf8").read()) for lang in LANGUAGES}
    kmedoids = KmedoidsClustering(languages)