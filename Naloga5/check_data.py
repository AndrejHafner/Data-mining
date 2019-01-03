import numpy as np
import pandas as pd

train_df = pd.read_csv("data/user_artists_training.dat",sep='\t')
test_df = pd.read_csv("data/user_artists_test.dat",sep='\t')


users_train = set(train_df['userID'])
users_test = set(test_df['userID'])

artist_train = set(train_df['artistID'])
artist_test = set(test_df['artistID'])

print(artist_test - artist_train)
print(len(artist_train & (artist_test - artist_train)))


# users intersection:1859
# users union:1892
# users test count:1859
# users train count:1892
#
# artist intersection:2902
# artist union:17632
# artist test count:4002
# artist train count:16532
# artist difference:1100


print()

print('users intersection:%d' % len(users_train & users_test))
print('users union:%d' % len(users_train | users_test))
print('users test count:%d' % len(users_test))
print('users train count:%d' % len(users_train))
print()
print('artist intersection:%d' % len(artist_train & artist_test))
print('artist union:%d' % len(artist_train | artist_test))
print('artist test count:%d' % len(artist_test))
print('artist train count:%d' % len(artist_train))

print('artist difference:%d' % len(artist_test - artist_train))



