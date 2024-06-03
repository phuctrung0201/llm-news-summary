import os

ARTICLES_PATH = "./dataset/articles"
SUMMARIES_PATH = "./dataset/summaries"


def load_dataset_path(limit=None):
    count = 0
    dataset_paths = []
    for topic in os.listdir(ARTICLES_PATH):
        for file in os.listdir(ARTICLES_PATH + "/" + topic):
            count += 1
            article_path = ARTICLES_PATH + "/" + topic + "/" + file
            summary_path = SUMMARIES_PATH + "/" + topic + "/" + file
            dataset_paths.append([article_path, summary_path])
            if limit != None and count >= limit:
                return dataset_paths
    return dataset_paths


def load_text(path):
    with open(path) as f:
        s = f.read()
        s = s.replace("\n", " ")
        s = s.replace("\'", "")
        s = s.replace("\"", "")
        s = s.replace(".", "")
        s = s.replace(",", "")
        s = s.replace("?", "")
        s = s.replace(":", "")
        s = s.replace("  ", " ")
        s = s.strip()
        return s
