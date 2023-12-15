import pickle
# import time
import pandas as pd
# import numpy as np
# import xgboost
# from sklearn.feature_extraction.text import CountVectorizer
def xgb_predictor(train_x):
    #######################################################################################################################
    # Implementing Text Classification  https://www.kaggle.com/code/meetnagadia/implementing-text-classification/notebook###
    extract_feature = False
    name_feature_pkl = 'count_vect.pkl'
    name_model_pkl = 'model.pkl'
    ###########################################################################################################################
    # #"""Count Vectors as features
    # Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document."""
    # create a count vectorizer object
    # start_time = time.time()
    if extract_feature:
        # count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        # count_vect.fit(trainDF['text'])
        # file = open(name_feature_pkl, 'wb')
        # pickle.dump(count_vect, file)
        # file.close()
        pass
    else:
        file = open(name_feature_pkl, 'rb')
        count_vect = pickle.load(file)
        file.close()

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x)
    # print("Count Vectors--- %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    feature_vector_train = xtrain_count.tocsc()
    if extract_feature:
        pass
        # classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        # classifier.fit(feature_vector_train, train_y)
        # file = open(name_model_pkl, 'wb')
        # pickle.dump(classifier, file)
        # file.close()
    else:
        file = open(name_model_pkl, 'rb')
        classifier = pickle.load(file)
        file.close()

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_train)

    # print("Count Vectors--- %s seconds ---" % (time.time() - start_time))
    # print("Xgb, Count Vectors: ",predictions[0])
    return predictions[0]

if __name__ == '__main__':
    trainDF = pd.read_csv('test.csv')
    # trainDF = df_new1.iloc[:110]
    # df_new1.columns = ['Body', 'label10']
    # print('test true label distribution:',df_new1.label10.value_counts())
    #
    # trainDF = df_new1[['Body', 'label10']]
    trainDF.columns = ['text', 'label']
    # trainDF.to_csv('./api/test.csv',index=False)
    train_x, train_y = [trainDF.iloc[0]['text']],  [trainDF.iloc[0]['label']]
    # print('train_y', np.unique(trainDF['label'], return_counts=True))
    for i, x in enumerate(trainDF['text'].values):
        # xgb_predictor([x])
        print (trainDF.iloc[i]['label'],xgb_predictor([x]))