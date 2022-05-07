from sklearn.datasets._base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

from tool import readfile, readbunchobj, writebunchobj


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功！！！")


if __name__ == '__main__':
    stopword_path = "训练Bunch/hlt_stop_words.txt"
    # bunch_path = "训练Bunch/bunch.dat"
    # space_path = "训练Bunch/tfdifspace.dat"
    # vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "测试Bunch/bunch.dat"
    space_path = "测试Bunch/tfdifspace.dat"
    train_tfidf_path = "训练Bunch/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
