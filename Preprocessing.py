import pandas as pd
import re
from tqdm import tqdm
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


class Preprocessing:
    def __init__(self, df=df):
        self.df = df.copy()

        self.clear_links()
        self.punctuation()
        self.drop_smth()
        # self.lemmatization()
        self.steammer()

    def check_missing(self):
        result = pd.concat([self.df.isnull().sum(), self.df.isnull().mean()], axis=1)
        result = result.rename(index=str, columns={0: "total missing", 1: "proportion"})
        return result

    def strip_punctuation(self, string):
        return re.sub(r"[^\w\s]", "", str(string).lower())

    def clear_links(self):
        self.df["msg_wo_links"] = self.df["message"].apply(
            lambda x: re.split("https:\/\/.*", str(x))
        )
        self.df = self.df[self.df["msg_wo_links"].notna()]

    def punctuation(self):
        self.df["clear_msg"] = self.df["msg_wo_links"].apply(self.strip_punctuation)

    def drop_smth(self):
        self.df = self.df.drop(
            [
                "channel_name",
                "channel_ID",
                "message_id",
                "sender_ID",
                "reply_to_msg_id",
                "time",
                "msg_wo_links",
                "message",
            ],
            axis=1,
        )

    def lemmatization(self):
        m = Mystem()
        self.df["clear_msg"].progress_apply(lambda x: m.lemmatize(x))

    def stemmatization(self, text):
        stop = stopwords.words("russian")
        stemmer = nltk.stem.snowball.RussianStemmer("russian")
        return " ".join(
            [stemmer.stem(word) for word in text.split(" ") if word not in stop]
        )

    def steammer(self):
        tqdm.pandas()
        self.df["clear_msg"] = self.df["clear_msg"].progress_apply(
            lambda x: self.stemmatization(x)
        )
        self.df.clear_msg = self.df.clear_msg.apply(lambda x: x.split(" "))

    def preproce_it(self):
        return self.df
