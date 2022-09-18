import os
import pandas as pd
from nltk.tokenize import sent_tokenize


def create_ViMs_csv(cluster_path, summary_path):
    """
    Input: The original ViMs dataset consisting of three folders: original, S3_summary, and summary
    Output: One csv file consisting of four columns of string: cluster_id, cluster, summary1, and summary2 structured as follow:
        + cluster_id: cluster folder name
        + cluster: sent 1<sent/>sent 2<sent/><doc/>sent 1<sent/>sent 2<sent/>sent 3<sent/><doc/>
        + summary1: sent 1<sent/>sent 2<sent/>
        + summary2: sent 1<sent/>sent 2<sent/>    
    """

    def extract_cluster(cluster_folder):
        cluster = []
        for file in os.listdir(cluster_folder):        
            doc = open(os.path.join(cluster_folder, file),encoding="utf8").readlines()[8:]            
            cluster.append([])
            for line in doc:
                cluster[-1] = cluster[-1] + sent_tokenize(line)                

        return "<doc/>".join(["<sent/>".join(doc) for doc in cluster])


    def get_summary(summary_file):
        summary = []
        for line in open(summary_file,encoding="utf8").readlines():
            summary = summary + sent_tokenize(line)
        return "<sent/>".join(summary)


    examples = []
    for cluster_name in os.listdir(cluster_path):        
        if cluster_name[0] == ".":
            continue                            
        
        cluster = extract_cluster(os.path.join(cluster_path, cluster_name, "original"))
        summary1 = get_summary(os.path.join(summary_path, cluster_name, "0.gold.txt"))
        summary2 = get_summary(os.path.join(summary_path, cluster_name, "1.gold.txt"))

        examples.append([cluster_name, cluster, summary1, summary2])

    df = pd.DataFrame(examples, columns=["cluster_id", "cluster", "summary1", "summary2"])
    df.to_csv("ViMs.csv",encoding="utf8", index=False)    
        

def read_ViMs_csv(file):
    """
    Input: The csv file outputted from the create_ViMs_csv function
    Output: The dataframe of four columns:
        + cluster_id: The cluster folder name
        + cluster: List of docs, each doc is a list of sents
        + summary1: List of sents in the summary created by the first annotator
        + summary2: List of sents in the summary created by the second annotator
    """

    df = pd.read_csv(file)
    df.cluster = df.cluster.apply(lambda cluster: [doc.split("<sent/>") for doc in cluster.split("<doc/>")])
    df.summary1 = df.summary1.apply(lambda summary: summary.split("<sent/>"))
    df.summary2 = df.summary2.apply(lambda summary: summary.split("<sent/>"))

    return df


def ViMs_statistic(df):
    """
    Input: The dataframe outputted by the read_ViMs_csv function
    Output: statistic of the dataset including:
        + nb_docs: number of documents in the cluster
        + longest_doc: the longest document in term of number of sentences
        + shortest_doc: the shortest document in term of number of sentences
        + longest_sent: the longest sentence in term of number of characters
        + shortest_sent: the shortest sentence in term of number of characters
    """

    print("dataset.shape:", df.shape)
    df["nb_docs"] = df.cluster.apply(lambda cluster: len(cluster))
    df["nb_sents"] = df.cluster.apply(lambda cluster: [len(doc) for doc in cluster])
    df["longest_doc (#sents)"] = df.nb_sents.apply(lambda nb_sents: max(nb_sents))
    df["shortest_doc (#sents)"] = df.nb_sents.apply(lambda nb_sents: min(nb_sents))
    df["longest_sent (#chars)"] = df.cluster.apply(lambda cluster: max([len(sent) for doc in cluster for sent in doc]))
    df["shortest_sent (#chars)"] = df.cluster.apply(lambda cluster: min([len(sent) for doc in cluster for sent in doc]))

    print(df.head())
    print(df.describe())

 
create_ViMs_csv("./original", "./summary")
df = read_ViMs_csv("./ViMs.csv")
# ViMs_statistic(df)
print(df["summary1"][0])
