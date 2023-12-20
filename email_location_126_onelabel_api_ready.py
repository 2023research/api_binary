import re, pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# import warnings
# warnings.filterwarnings('ignore')

#############################################################################################################
training = False
name_feature_pkl = './api/count_vect_cate.pkl'
name_model_pkl = './api/model_cate.pkl'

#################################################################################################################

def cate_predictor(email):
    # remove stop words
    #Remove Stopwords
    stop_words = set(stopwords.words('english'))
    # function to remove stopwords
    def remove_stopwords(text):
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)
    #Clean Text
    def clean_text(text):
        text = text.lower()
        text = re.sub("[^a-zA-Z]"," ",text)
        text = ' '.join(text.split())
        return text

    #stemming
    stemmer = SnowballStemmer("english")
    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    email=remove_stopwords(email)
    email=clean_text(email)
    email=stemming(email)

    # train_data['Text'] = train_data['Text'].apply(lambda x: remove_stopwords(x))
    # train_data['Text'] = train_data['Text'].apply(lambda x:clean_text(x))
    # train_data['Text'] = train_data['Text'].apply(stemming)

    ####################################################################################################################
    # split dataset into training and validation set

    # binary_labels = train_data.iloc[:,1:]
    # xtrain, xval, ytrain, yval = train_test_split(train_data['Text'], binary_labels, test_size=0.001, random_state=9)
    # print ('valid size:',len(yval))

    xval = [email]

    if training:
        # tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
        # xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
        # cate_names = binary_labels.columns
        # file = open(name_feature_pkl, 'wb')
        # pickle.dump([tfidf_vectorizer,cate_names], file)
        # file.close()
        pass
    else:
        file = open(name_feature_pkl, 'rb')
        tfidf_vectorizer, cate_names = pickle.load(file)
        file.close()
    xval_tfidf = tfidf_vectorizer.transform(xval)

    # Using Gaussian Naive Bayes
    # from skmultilearn.problem_transform import BinaryRelevance
    # from sklearn.naive_bayes import GaussianNB

    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    if training:
        # classifier = BinaryRelevance(GaussianNB())
        # classifier.fit(xtrain_tfidf, ytrain)
        # file = open(name_model_pkl, 'wb')
        # pickle.dump(classifier, file)
        # file.close()
        pass
    else:
        file = open(name_model_pkl, 'rb')
        classifier = pickle.load(file)
        file.close()


    # predict
    predictions = classifier.predict(xval_tfidf)
    pred = predictions.toarray()
    pred = cate_names[pred[0]==1]
    # print (pred[0])
    return pred.values
if __name__ == '__main__':

    df=pd.read_csv('./results/126_cate_data.csv')
    # train_data = df[['Body']]#.merge(binary_labels, how='inner', left_index=True, right_index=True)
    # train_data.rename(columns={'Body': 'Text'}, inplace=True)
    # # # label count
    # x = train_data.iloc[:, 1:].sum()
    # rowsums = train_data.iloc[:, 1:].sum(axis=1)
    # no_label_count = 0
    # for sum in rowsums.items():
    #     if sum == 0:
    #         no_label_count += 1
    #
    # print("Total number of samples = ", len(train_data))
    # print("Total number of articles without label = ", no_label_count)
    # print("Total labels = ", x.sum())
    for email in df['Body'].values:
        pred = cate_predictor(email)
        print (pred)