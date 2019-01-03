import numpy as np
import pandas as pd
import math

K = 10 # in netflix usually 50
ALPHA = 0.004 # learning rate
LAMBDA = 0.02 # regularization factor best 0.03
USER_CNT = 1892
ARTIST_CNT = 17632
TRAIN_SET_FRACTION = 0.8
ITERATIONS = 100


class ArtistRecommenderSystem(object):

    def __init__(self, df,K,alpha,lambda_,artists_with_tags):
        train_df, validation_df = np.split(df.sample(frac=1,random_state=42),[int(TRAIN_SET_FRACTION * len(df))])
        self.train_df = train_df.reset_index(drop=True)
        self.validation_df = validation_df.reset_index(drop=True)
        self.K = K
        self.alpha = alpha
        self._lambda = lambda_
        self.artists_with_tags = artists_with_tags
        self.train_artists_set = set(train_df['artistID'])
        self.user_max_idx = int(np.max(df['userID']))
        self.artist_max_idx = int(np.max(df['artistID']))
        self.train_model()

    def __call__(self,user_id,artist_ID):
        if artist_ID not in self.train_artists_set:
            success,prediction = self.get_weighted_tag_prediction(artist_ID,user_id)
            if success:
                return prediction
            else:
                return self.get_prediction(user_id,artist_ID)
        return self.get_prediction(user_id,artist_ID)

    def train_model(self):

        # Initialize the R matrix
        # self.create_rating_matrix(train_df)

        # Initialize P and Q matrix
        self.init_PQ()

        # Initialize biases
        self.bias_user = np.zeros(self.user_max_idx + 1)
        self.bias_artist = np.zeros(self.artist_max_idx + 1)
        self.bias = np.average(self.train_df['weight'])

        # Transpose for calculation
        self.Q = self.Q.T

        prev_rmse = 1000
        rmse = 100
        step = 1
        # Iterations until convergence
        while(rmse < prev_rmse):
            prev_rmse = rmse
            for idx,row in self.train_df.iterrows():

                user_id = int(row['userID'])
                artist_id = int(row['artistID'])
                rating = row['weight']

                # Calculate the error
                prediction = self.get_prediction(user_id,artist_id)
                eui = rating - prediction

                # Update biases
                self.bias_user[user_id] = self.bias_user[user_id] + self.alpha * (eui - self._lambda * self.bias_user[user_id])
                self.bias_artist[artist_id] = self.bias_artist[artist_id] + self.alpha * (eui - self._lambda * self.bias_artist[artist_id])

                # Update P and Q matrices in the direction of the gradient
                self.P[user_id,:] = self.P[user_id,:] + self.alpha * (eui * self.Q[artist_id,:] - self._lambda * self.P[user_id,:])
                self.Q[artist_id,:] = self.Q[artist_id,:] + self.alpha * (eui * self.P[user_id,:] - self._lambda * self.Q[artist_id,:])

            # Calculate RMSE
            rmse = self.rmse(self.validation_df)
            print("Iteration %d: rmse=%.4f" % (step,rmse))
            step += 1
        print('Finished, started to overfit.')



        # Transpose back
        #self.Q = self.Q.T

    def get_prediction(self,user_id,artist_id):
        return self.bias + self.bias_user[user_id] + self.bias_artist[artist_id] + self.P[user_id,:].dot(self.Q[artist_id,:].T)

    def get_weighted_tag_prediction(self,artist_id,user_id):
        artist_tag = self.artists_with_tags[self.artists_with_tags['artistID'] == artist_id]
        if len(artist_tag) == 0:
            return False,-1
        same_tag_artists = self.artists_with_tags[self.artists_with_tags['tagID'] == int(artist_tag['tagID'])]
        tag_count_sum = np.sum(same_tag_artists['tagCount'])

        prediction = 0
        # Get the weighted prediction of the user for all the songs with this tag
        for idx,row in same_tag_artists.iterrows():
            prediction += float(self.get_prediction(user_id,int(row['artistID']))) * float(row['tagCount'] / tag_count_sum)
        return True, prediction

    def init_PQ(self):
        self.P = np.random.normal(scale=1./self.K,size=(self.user_max_idx + 1,self.K)) * 0.1
        self.Q = np.random.normal(scale=1./self.K,size=(self.K,self.artist_max_idx + 1)) * 0.1

    def rmse(self,df):
        error = 0
        for idx, row in df.iterrows():
            user_id = int(row['userID'])
            artist_id = int(row['artistID'])
            rating = row['weight']
            predicted = self.get_prediction(user_id,artist_id)
            error += pow(predicted - rating,2)
        return np.sqrt(error / len(df))

# Creates a file that has artistIDs with the most frequent tag and its count
def get_artist_tags():
    tags = pd.read_csv("data/user_taggedartists.dat", sep='\t')
    tagged_artists = set(tags['artistID'])  # 12513 tagged artists
    artist_with_tags = pd.DataFrame(columns=['artistID', 'tagID', 'tagCount'])
    for id in tagged_artists:
        artist_tags = tags[tags['artistID'] == id]
        (values, counts) = np.unique(artist_tags['tagID'], return_counts=True)
        most_tagged_idx = np.argmax(counts)
        most_tagged = values[most_tagged_idx]
        most_tagged_cnt = counts[most_tagged_idx]
        artist_with_tags = artist_with_tags.append({'artistID': id, 'tagID': most_tagged, 'tagCount': most_tagged_cnt},
                                                   ignore_index=True)
    artist_with_tags.to_csv('data/artists_most_tagged.dat', sep='\t', encoding='utf-8', index=False)


def test_model(model,df,filename='predictions/prediction.txt'):
    test_values = df.values
    print('Outputing to a file: %s' % filename)
    with open(filename,'w') as f:
        for i in range(len(test_values)):
            prediction = model(test_values[i,0],test_values[i,1])
            print(prediction,file=f)



if __name__ == "__main__":
    train_df = pd.read_csv("data/user_artists_training.dat",sep='\t')
    test_df = pd.read_csv("data/user_artists_test.dat",sep='\t')
    artists_with_tags = pd.read_csv('data/artists_most_tagged.dat', sep='\t') # Run get_artis_tags() before to create the file
    rec_sys = ArtistRecommenderSystem(train_df,K,ALPHA,LAMBDA,artists_with_tags)

    #TODO pred oddajo MORA IZPISATI NA STDOUT!
    test_model(rec_sys,test_df,filename='predictions/3_prediction_with_bias.txt')

