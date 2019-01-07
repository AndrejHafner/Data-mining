import numpy as np
import pandas as pd

K = 12 # in netflix usually 50
ALPHA = 0.003 # learning rate 0.004
LAMBDA = 0.03 # regularization factor best 0.03
USER_CNT = 1892
ARTIST_CNT = 17632
TRAIN_SET_FRACTION = 0.9
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
        self.train_artists_set = set(self.train_df['artistID'])
        self.user_max_idx = int(np.max(df['userID']))
        self.artist_max_idx = int(np.max(df['artistID']))
        self.matrix_factorization()

    def __call__(self,user_id,artist_ID):
        if artist_ID not in self.train_artists_set:
            success,prediction = self.get_weighted_tag_prediction(artist_ID,user_id)
            if success:
                return prediction
            else:
                return self.get_prediction(user_id,artist_ID)
        return self.get_prediction(user_id,artist_ID)

    def matrix_factorization(self):

        # Initialize P and Q matrix
        self.init_PQ()

        # Transpose for calculation
        self.Q = self.Q.T

        # Initialize user and artist biases
        self.P[:,self.K-2] = 1
        self.Q[:,self.K-1] = 1

        prev_rmse = 1000
        rmse = 100
        step = 1
        # Iterations until convergence
        while(rmse < prev_rmse):
            prev_rmse = rmse
            for idx,row in self.train_df.iterrows():

                user_id = int(row['userID'])
                artist_id = int(row['artistID'])
                rating = row['weight'] #- self.bias_user[user_id] - self.bias_artist[artist_id] - self.avg_rating

                # Calculate the error
                prediction = self.get_prediction(user_id,artist_id) #self.P[user_id,:].dot(self.Q[artist_id,:].T)
                eui = (1/np.sqrt(2)) * (rating - prediction)

                # Update P and Q matrices in the direction of the gradient
                self.P[user_id, 0:K-2] = self.P[user_id, 0:K-2] + self.alpha * (eui * self.Q[artist_id, 0:K-2] - self._lambda * self.P[user_id, 0:K-2])
                self.Q[artist_id, 0:K-2] = self.Q[artist_id, 0:K-2] + self.alpha * (eui * self.P[user_id, 0:K-2] - self._lambda * self.Q[artist_id, 0:K-2])

                self.P[user_id,K-1] = self.P[user_id, K-1] + self.alpha * (eui - self._lambda * self.P[user_id, K-1])
                self.Q[artist_id,K-2] = self.Q[artist_id, K-2] + self.alpha * (eui - self._lambda * self.Q[artist_id, K-2])
            # Calculate RMSE
            rmse = self.rmse(self.validation_df)
            # print("Iteration %d: rmse=%.4f" % (step,rmse))
            step += 1
        # print('Finished, started to overfit.')


    def get_prediction(self,user_id,artist_id):
        return self.P[user_id,:].dot(self.Q[artist_id,:].T)

    def get_weighted_tag_prediction(self,artist_id,user_id):
        artist_tag = self.artists_with_tags[self.artists_with_tags['artistID'] == artist_id]
        if len(artist_tag) == 0:
            return False,-1
        same_tag_artists = self.artists_with_tags[self.artists_with_tags['tagID'] == int(artist_tag['tagID'])]
        tag_count_sum = np.sum(same_tag_artists['tagCount'])
        prediction_sum = 0

        # # Get the weighted prediction of the user for all the songs with this tag
        for idx,row in same_tag_artists.iterrows():
            prediction_sum += self.get_prediction(user_id,int(row['artistID']))
        return True, (prediction_sum / len(same_tag_artists))

    def init_PQ(self):
        self.P = np.random.normal(scale=1./self.K,size=(self.user_max_idx + 1,self.K)) * 0.1
        self.Q = np.random.normal(scale=1./self.K,size=(self.K,self.artist_max_idx + 1)) * 0.1

    def init_biases(self):
        self.avg_rating = np.average(self.train_df['weight'])
        self.bias_user = np.zeros(self.user_max_idx + 1)
        self.bias_artist = np.zeros(self.artist_max_idx + 1)

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

# Gets all of the users friends into a dictionary
def get_user_friends(df):
    users_set = set(df['userID'])
    return {user_id: [ friend_id for friend_id in df[df['userID'] == user_id].values[:,1] ] for user_id in users_set}

# Tests the model and outputs the result to a file
def test_model(model,df,filename='predictions/prediction.txt'):
    test_values = df.values
    print('Outputing to a file: %s' % filename)
    with open(filename,'w') as f:
        for i in range(len(test_values)):
            prediction = model(test_values[i,0],test_values[i,1])
            print(prediction,file=f)

def test_and_output_to_stdout(model,df):
    test_values = df.values
    for i in range(len(test_values)):
        prediction = model(test_values[i, 0], test_values[i, 1])
        print(prediction)


if __name__ == "__main__":
    train_df = pd.read_csv("user_artists_training.dat",sep='\t')
    test_df = pd.read_csv("user_artists_test.dat",sep='\t')
    artists_with_tags = pd.read_csv('artists_most_tagged.dat', sep='\t') # Run get_artist_tags() before to create the file
    # user_friends = pd.read_csv('data/user_friends.dat', sep='\t')
    # friends_dict = get_user_friends(user_friends)
    rec_sys = ArtistRecommenderSystem(train_df,K,ALPHA,LAMBDA,artists_with_tags)

    test_and_output_to_stdout(rec_sys,test_df)
    # test_model(rec_sys,test_df,filename='predictions/3_prediction_with_bias.txt')

