import numpy as np
import pandas as pd
train_df = pd.read_csv("data/user_artists_training.dat",sep='\t')
test_df = pd.read_csv("data/user_artists_test.dat",sep='\t')

#
# users_train = set(train_df['userID'])
# users_test = set(test_df['userID'])
#
# artist_train = set(train_df['artistID'])
# artist_test = set(test_df['artistID'])
# test_df = pd.read_csv("data/user_artists_test.dat", sep='\t')
artists_with_tags = pd.read_csv('data/artists_most_tagged.dat',sep='\t')  # Run get_artis_tags() before to create the file

#


artist_tag = artists_with_tags[artists_with_tags['artistID'] == 5456]
same_tag_artists = artists_with_tags[artists_with_tags['tagID'] == int(artist_tag['tagID'])]

tag_count_sum = np.sum(same_tag_artists['tagCount'])


print(np.sum(same_tag_artists['tagCount']))


k =0
# print(len(test_df[test_df['artistID'].isin((artist_test - artist_train) & set(artists_with_tags['artistID']))]))
#
# train_artists_set = set(train_df['artistID'])
#
# for idx,row in test_df.iterrows():
#     if row['artistID'] not in train_artists_set:
#         print(row['artistID'])


def get_artist_tags():
    tags = pd.read_csv("data/user_taggedartists.dat", sep='\t')

    tagged_artists = set(tags['artistID']) # 12513 tagged artists
    artist_with_tags = pd.DataFrame(columns=['artistID','tagID','tagCount'])
    for id in tagged_artists:
        artist_tags = tags[tags['artistID'] == id]
        (values, counts) = np.unique(artist_tags['tagID'], return_counts=True)
        most_tagged_idx = np.argmax(counts)
        most_tagged = values[most_tagged_idx]
        most_tagged_cnt = counts[most_tagged_idx]
        artist_with_tags = artist_with_tags.append({'artistID':id,'tagID':most_tagged,'tagCount':most_tagged_cnt},ignore_index=True)
    artist_with_tags.to_csv('data/artists_most_tagged.dat',sep='\t',encoding='utf-8',index=False)


# train_df = pd.read_csv("data/user_artists_training.dat",sep='\t')
# test_df = pd.read_csv("data/user_artists_test.dat",sep='\t')
#
#
# users_train = set(train_df['userID'])
# users_test = set(test_df['userID'])
#
# artist_train = set(train_df['artistID'])
# artist_test = set(test_df['artistID'])


#
# print(len(set(tags['artistID'])))
#
# print(len(set(tags['artistID']) & (artist_test - artist_train))) # 568/1100 artists missing from training data are tagged