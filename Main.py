import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib 
matplotlib.use('TkAgg') #pip install --upgrade matplotlib
from sklearn.preprocessing import*
from sklearn.cluster import*
from sklearn.metrics import*
from functools import*
from sklearn.manifold import*
from scipy.spatial.distance import*
from sklearn.metrics.pairwise import*
from scipy.spatial.distance import*
from sklearn.decomposition import*
from scipy.cluster.hierarchy import*
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import csv
from scipy.spatial.distance import jaccard
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras import regularizers


#define the paths
# Get the path of the current working directory
cwd = os.getcwd()

# Define the filenames
filename1 = 'Dataset.npy'
filename2 = 'Dataset.csv'
filename3 = 'Dataset_updated.csv'

# Get the full paths of the files
path1 = os.path.join(cwd, filename1)
path2 = os.path.join(cwd, filename2)
path3 = os.path.join(cwd, filename3)


def ExistFile(x):
    if (os.path.exists(os.path.join(cwd, x))): 
        return True
    else: 
        return False

def read_clean_data(case):
    print("\n")
    if (case=='Dataset_updated.csv'):
        df = pd.read_csv(path3)
    elif(case=='Dataset.csv'):
        df = pd.read_csv(path2)
            # Split the single column into four columns
        df[['User', 'Movie', 'Rate', 'Time']] = df['0'].str.split(',', expand=True)
            # Drop the original single column
        df.drop('0', axis=1, inplace=True)
            # Preprocess the data by removing the 'ur' prefix from the User column
        df['User'] = df['User'].apply(lambda x: x[2:])
            # Preprocess the data by removing the 'tt' prefix from the User column
        df['Movie'] = df['Movie'].apply(lambda x: x[2:])
            # Convert the Time column to the "DD/MM/YYYY" format
        Times=df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%Y/%m/%d')
        Users=df['User']
        Users_uni=df['User'].unique()
        Movies=df['Movie']
        Movies_uni=df['Movie'].unique()
        Rates=df['Rate'].astype(int)

    # Save the updated DataFrame as a new CSV file
        df.to_csv(path3, index=False)
    elif(case=='Dataset.npy'):
            # Load the data from the .npy file
        data = np.load(path1)
            # Create a DataFrame from the data
        dfSave = pd.DataFrame(data)
            # Save the DataFrame as a CSV file
        dfSave.to_csv(path2, index=False)
        df = pd.read_csv(path2)
            # Split the single column into four columns
        df[['User', 'Movie', 'Rate', 'Time']] = df['0'].str.split(',', expand=True)
            # Drop the original single column
        df.drop('0', axis=1, inplace=True)
            # Preprocess the data by removing the 'ur' prefix from the User column
        df['User'] = df['User'].apply(lambda x: x[2:])
        User = df['User'].unique()
            # Preprocess the data by removing the 'tt' prefix from the User column
        df['Movie'] = df['Movie'].apply(lambda x: x[2:])
        Movie= df['Movie'].unique()
            # Convert the Time column to the "DD/MM/YYYY" format
        Times=df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%Y/%m/%d')
        Users=df['User']
        Users_uni=df['User'].unique()
        Movies=df['Movie']
        Movies_uni=df['Movie'].unique()
        Rates=df['Rate'].astype(int)

    # Save the updated DataFrame as a new CSV file
        df.to_csv(path3, index=True)

def read_the_rows():
    try:
        rows = int(input("Enter the number of rows to read: (0->All data) (empty->default value:5000): "))
        if (rows==0): return None
    except ValueError:
        return 5000
    return rows

def loading_data():
    if ExistFile('Dataset_updated.csv'):
        print("Read from Dataset_updated.csv")
        read_clean_data('Dataset_updated.csv')  
    elif ExistFile('Dataset.csv'):
        print("Read from Dataset.csv and Create Dataset_updated.csv")
        read_clean_data('Dataset.csv')       
    else:
        print("Read from Dataset.npy, Create Dataset.csv and Dataset_Updated.csv")
        read_clean_data('Dataset.npy')

def question():
    try:
        print('1->Προ-επεξεργασία Δεδομένων\n')
        print('2,3->Αλγόριθμοι Ομαδοποίησης Δεδομένων\n')
        print('4,5->Αλγόριθμοι Παραγωγής Συστάσεων με Χρήση Τεχνητών Νευρωνικών Δικτύων\n')
        answer = int(input("Enter the number of Question you want to get the answer : "))
        return answer
    except ValueError:
        return None
    

def jaccard_distances(array):
    n_users, n_movies = array.shape
    distances = np.zeros((n_users, n_users))

    for u in range(n_users):
        for v in range(u+1, n_users):
            u_rated = set(np.where(array[u, :] > 0)[0])
            v_rated = set(np.where(array[v, :] > 0)[0])
            intersection = len(u_rated & v_rated)
            union = len(u_rated | v_rated)
            if union == 0:
                distances[u, v] = 1
            else:
                distances[u, v] = 1 - (intersection / union)
            distances[v, u] = distances[u, v]
            #print(distances)

    return distances

    

def optimal_clusters(metric, array, min_clusters=2, max_clusters=10):
    array_nonzero = array[array != 0]
    array_reshaped = np.column_stack(np.where(array != 0))

    if metric == 'cosine':
        distances = cosine_distances(array_reshaped)
    elif metric == 'euclidean':
        distances = euclidean_distances(array_reshaped)
    elif metric == 'jaccard':
        distances = jaccard_distances(array)
    else:
        raise ValueError("Invalid metric specified. Supported metrics: 'cosine', 'euclidean', 'jaccard'")

    cluster_range = range(min_clusters, max_clusters + 1)
    silhouette_scores = []

    for n_clusters in cluster_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        clustering.fit(distances)
        labels = clustering.labels_
        silhouette_avg = silhouette_score(distances, labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)

    best_cluster_num = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    return best_cluster_num

def make_plot(metric, cluster, array):
    # Remove zero values from the array
    array_nonzero = array[array != 0]

    # Reshape the array into a 2D format
    array_reshaped = np.column_stack(np.where(array != 0))
    
    # Create a mapping from original indices to reshaped indices
    index_mapping = {tuple(idx): i for i, idx in enumerate(np.column_stack(np.where(array != 0)))}

    # Calculate distances based on the specified metric
    if metric == 'cosine':
        distances = cosine_distances(array_reshaped)
        clustering = AgglomerativeClustering(n_clusters=cluster, affinity='precomputed', linkage='average')
        clustering.fit(distances)
        plt.title(f'{metric.capitalize()} distance with {cluster} clusters')
    elif metric == 'euclidean':
        distances = euclidean_distances(array_reshaped)
        clustering = AgglomerativeClustering(n_clusters=cluster, affinity='precomputed', linkage='average')
        clustering.fit(distances)
        plt.title(f'{metric.capitalize()} distance with {cluster} clusters')
    elif metric == 'jaccard':
        distances = jaccard_distances(array)
        clustering = AgglomerativeClustering(n_clusters=cluster, affinity='precomputed', linkage='average')
        clustering.fit(distances)
        plt.title(f'Jaccard distance with {cluster} clusters')
    else:
        raise ValueError("Invalid metric specified. Supported metrics: 'cosine', 'euclidean'")

    # Create a labels array with the same shape as array_reshaped
    reshaped_labels = np.full(array_reshaped.shape[0], -1)
    for i, label in enumerate(clustering.labels_):
        reshaped_labels[index_mapping[tuple(array_reshaped[i])]] = label

    # Plot the points with different colors for each cluster
    plt.scatter(array_reshaped[:, 0], array_reshaped[:, 1], c=reshaped_labels, cmap='viridis')

    # Set plot title and show the plot
    plt.show()

def train_faction(user):
    df = pd.read_csv(path3, nrows=read_the_rows())
    Rmin = 45
    Rmax = 50
    # Group by and filter dataframe based on number of ratings per user
    user_counts = df.groupby('User').count()['Movie']
    user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
    df_filtered = df[df['User'].isin(user_counts_filtered.index)]

    # Print number of users with ratings between Rmin and Rmax
    num_users_filtered = len(user_counts_filtered)
    print(f"Number of users with ratings between {Rmin} and {Rmax}: {num_users_filtered}")
    # print(user_counts_filtered)

    # Create a list of unique users and movies
    User = df_filtered['User'].unique()
    Movie = df_filtered['Movie'].unique()

    # Create a 2D numpy array filled with zeros
    R = np.zeros((len(User), len(Movie)))

    # Loop through the filtered dataframe and populate the R array with ratings
    for i, row in df_filtered.iterrows():
        user_index = np.where(User == row['User'])[0][0]
        movie_index = np.where(Movie == row['Movie'])[0][0]
        R[user_index, movie_index] = row['Rate']

    # Define the number of nearest k-1 neighbors to consider
    k = 31

    # Compute the Jaccard similarity matrix for all users in the filtered dataset
    simularity_matrix = 1 - jaccard_distances(R)

    # Create a list to store the top k users for each user in the filtered dataset
    neighbor_list = []

    print(simularity_matrix)
    print(jaccard_distances(R))
    print('/n')

    id_answer_list = []
    id_dictionary = {user: {} for user in User}
    # Loop over each user in the filtered dataset
    for i, user in enumerate(User):
        # Get the row of the Jaccard similarity matrix corresponding to the current user
        jaccard_row = simularity_matrix[i, :]

        # Sort the Jaccard similarity row in descending order and get the indices of the top k users
        topk_indices = np.argsort(-jaccard_row)[:k]

        # Get the user IDs of the top k users (excluding the current user)
        topk_users = [User[idx] for idx in topk_indices if User[idx] != user]

        # Store the top k users in the neighbor list for the current user
        neighbor_list.append(topk_users)

        id_answer_list.append(user)
        id_dictionary[user] = {'user': user, 'neighbor_list': neighbor_list}
        neighbor_list = []

    # Loop over each user in the answer list
    for user in id_answer_list:
        # Print the neighbor list for the current user
        print(f"Neighbor list for user {user}: {id_dictionary[user]['neighbor_list']}")

    print(id_answer_list)
    ratings_answers = []
    for user in id_answer_list:
        # Find the index of the user in the User array
        user_index = np.where(User == user)[0][0]

        # Access the row of the R array corresponding to the user's index to get the ratings for all movies
        user_ratings = R[user_index, :]
        ratings_answers.append(user_ratings)

    ratings_dictionary = {user: {} for user in User}
    for user_id, user_data in id_dictionary.items():
        user_ratings = []
        for neighbor_list in user_data['neighbor_list']:
            for neighbor in neighbor_list:
                neighbor_index = np.where(User == neighbor)[0][0]
                neighbor_ratings = R[neighbor_index, :]
                ratings_dictionary[user_id][neighbor] = neighbor_ratings

    for user_id, user_data in ratings_dictionary.items():
        neighbor_ratings = user_data.values()
        print(f"Ratings of neighbors for user {user_id}: {list(neighbor_ratings)}")

    # Select a random user from the dataframe
    input_user = int(user)

    # Print the selected user
    print("Randomly selected user:", input_user)

    # Check if input user exists in dataset
    if input_user not in User:
        print("User not found in dataset.")
    else:
        # Find the index of the user in the User array
        user_index = np.where(User == input_user)[0][0]

        # Access the row of the R array corresponding to the user's index to get the ratings for all movies
        user_ratings = R[user_index, :]

        # Print the ratings of the input user for all movies
        print(f"Ratings of user {input_user} for all movies: {user_ratings}")

        ratings_dictionary = {user: {} for user in User}

        for user_id, user_data in id_dictionary.items():
            user_ratings = []
            for neighbor_list in user_data['neighbor_list']:
                for neighbor in neighbor_list:
                    neighbor_index = np.where(User == neighbor)[0][0]
                    neighbor_ratings = R[neighbor_index, :]
                    ratings_dictionary[user_id][neighbor] = neighbor_ratings

        train_ratings_dict = {}
        test_ratings_dict = {}

        # Loop over each user and their neighbor ratings
        for user_id, user_data in ratings_dictionary.items():
            # Collect the neighbor ratings into a list
            neighbor_ratings_list = list(user_data.values())

            # Combine the neighbor ratings into a 2D numpy array
            input_features = np.array(neighbor_ratings_list)

            # Create an array of actual ratings for the input user, based on the ratings_answers dictionary
            actual_rating = np.array([ratings_answers[np.where(User == user_id)[0][0]]] * len(input_features))

            # Split the input features and actual ratings into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(
                input_features, actual_rating, test_size=0.2, random_state=42
            )

            # Create a dictionary containing the training features and labels for the current user
            train_ratings_dict[user_id] = {'input_features': train_features, 'labels': train_labels}

            # Create a dictionary containing the testing features and labels for the current user
            test_ratings_dict[user_id] = {'input_features': test_features, 'labels': test_labels}

        # Combine the training features and labels for all users into a single dataframe and numpy array, respectively
        x_train = pd.concat(
            [pd.DataFrame(train_ratings_dict[user_id]['input_features']) for user_id in train_ratings_dict.keys()],
            ignore_index=True)
        y_train = np.concatenate([train_ratings_dict[user_id]['labels'] for user_id in train_ratings_dict.keys()])

        # Combine the testing features and labels for all users into a single dataframe and numpy array, respectively
        x_test = pd.concat(
            [pd.DataFrame(test_ratings_dict[user_id]['input_features']) for user_id in test_ratings_dict.keys()],
            ignore_index=True)
        y_test = np.concatenate([test_ratings_dict[user_id]['labels'] for user_id in test_ratings_dict.keys()])


        model = Sequential()
        model.add(Embedding(len(Movie) + 1, 16, input_length=x_train.shape[1]))
        model.add(Flatten())
        model.add(Dense(len(Movie), activation='linear'))
        model.add(Dense(len(Movie), activation='sigmoid'))
        model.add(Dense(len(Movie), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'mean_squared_error'])

        model.fit(x_train, y_train, epochs=100, batch_size=32)
        y_hat = model.predict(x_test)
        print(y_hat)
        print(y_test)


def solution():
    df = pd.read_csv(path3,nrows=read_the_rows())
    y=question()
    if(y==0 or y is None):
        #1)Να βρείτε το σύνολο των μοναδικών χρηστών 𝑼 και το σύνολο των μοναδικών αντικειμένων 𝑰. ------------------
        User = df['User'].unique()
        Movie= df['Movie'].unique()
        print(f"Number of unique users: {len(User)}")
        print(f"Number of unique items: {len(Movie)}\n")

        #2 Ερώτημα---------------------------------------------------------------------------------------------------
        # Define the minimum and maximum number of ratings per user
        Rmin = 10
        Rmax = 50
        # Find the set of users with the required number of ratings
        print("Όλες οι Βαθμολογίες χρηστών ")
        user_counts = df.groupby('User').count()['Movie']
        print(user_counts)
        print("\n")
        # Filter the user_counts array based on the condition Rmin <= count <= Rmax
        print("Όλες οι Βαθμολογίες χρηστών μεταξύ "+ f"{Rmin}"+" και "+f"{Rmax}")
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        print(user_counts_filtered) 
        print("\n")

        #3 Ερώτημα --------------------------------------------------------------------------------------------------
        #Group the data by User and get the minimum and maximum Time for each group
        user_time = df.groupby('User')['Time'].agg(['min', 'max'])
        # print(user_time) 
        # print("\n")

        # convert the 'min' and 'max' columns to a datetime format
        user_time['min'] = pd.to_datetime(user_time['min'])
        user_time['max'] = pd.to_datetime(user_time['max'])
        # calculate the range of time in days
        user_time['Range'] = (user_time['max'] - user_time['min']).dt.days
        #print the result
        print(user_time)
        print("\n")
        #ploting the results
        plt.hist(user_time['Range'],color='blue')

        # ονομασία άξονα X
        plt.xlabel('User')
        # ονομασία άξονα Y
        plt.ylabel('Range (days)')
        plt.show()

        # Plot the histogram
        plt.hist(user_counts_filtered)
        plt.title("Histogram of User Ratings")
        plt.xlabel("Ratings")
        plt.ylabel("Frequency")
        plt.show()

        #4 Ερώτημα-------------------------------------------------------------------------------
        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie))) 

        # Loop through the dataframe and populate the R array with ratings
        for i, row in df.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']
        print(R)
        print("\n")

        try:
            answer = int(input("Enter the number of Cluster(s) - L you want to get the Plot (Empty->Best Elbow method): "))
        except ValueError:
            answer = None
        if answer is None:
            answer=optimal_clusters('euclidean',R)
            print("Euclidean clusters Elbow=" + str(answer))
            make_plot('euclidean',answer,R)
            
            answer=optimal_clusters('cosine',R)
            print("Cosine clusters Elbow=" + str(answer))
            make_plot('cosine',answer,R)
        else:
           print("Euclidean clusters=" + str(answer))
           make_plot('euclidean',answer,R)
           print("Cosine clusters=" + str(answer))
           make_plot('cosine',answer,R) 
        try:
            answer = int(input("Enter the number of Cluster(s) - L you want to get the Plot for jaccard distance (Empty->Best Elbow method): "))
        except ValueError:
            answer = None
        if answer is None:            
            answer=optimal_clusters('jaccard',R)
            print("Jaccard clusters Elbow=" + str(answer))
            make_plot('jaccard', answer, R)
        else:
            print("Jaccard clusters=" + str(answer))
            make_plot('jaccard', answer, R)


    elif(y==1):

        print('Προ-επεξεργασία Δεδομένων \n')
        #1 Ερώτημα-----------------------------------------------------------------------------------------------------
        print('1o Ερώτημα')
        User = df['User'].unique()
        Movie= df['Movie'].unique()
        print(f"Number of unique users: {len(User)}")
        print(f"Number of unique items: {len(Movie)}\n")

        #2 Ερώτημα---------------------------------------------------------------------------------------------------
        print('2o Ερώτημα')
        # Define the minimum and maximum number of ratings per user
        Rmin = 10
        Rmax = 50
        # Find the set of users with the required number of ratings
        print("Όλες οι Βαθμολογίες χρηστών ")
        user_counts = df.groupby('User').count()['Movie']
        print(user_counts)
        print("\n")
        # Filter the user_counts array based on the condition Rmin <= count <= Rmax
        print("Όλες οι Βαθμολογίες χρηστών μεταξύ "+ f"{Rmin}"+" και "+f"{Rmax}")
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        print(user_counts_filtered) 
        print("\n")

        user_counts = df.groupby('User').count()['Movie']
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        #3 Ερώτημα --------------------------------------------------------------------------------------------------
        print('3o Ερώτημα')
        #Group the data by User and get the minimum and maximum Time for each group
        user_time = df.groupby('User')['Time'].agg(['min', 'max'])
        # print(user_time) 
        # print("\n")

        # convert the 'min' and 'max' columns to a datetime format
        user_time['min'] = pd.to_datetime(user_time['min'])
        user_time['max'] = pd.to_datetime(user_time['max'])

        # calculate the range of time in days
        user_time['Range'] = (user_time['max'] - user_time['min']).dt.days

        #print the result
        print(user_time)
        print("\n")

        #ploting the results
        plt.hist(user_time['Range'],color='blue')

        # ονομασία άξονα X
        plt.xlabel('User')
        # ονομασία άξονα Y
        plt.ylabel('Range (days)')
        # ονομασία άξονα X
        # plt.xticks(user_time.index)
        # εμφάνιση του γραφήματος
        plt.show()
        # Plot the histogram
        plt.hist(user_counts_filtered)
        plt.title("Histogram of User Ratings")
        plt.xlabel("Ratings")
        plt.ylabel("Frequency")
        plt.show()
        #Δημιουργία πίνακα R[Χρήστης, Ταινία] = Βαθμολογία 
        User = df['User'].unique()
        Movie= df['Movie'].unique()

        #4 Ερώτημα-------------------------------------------------------------------------------
        print('4o Ερώτημα')
        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie))) 

        # Loop through the dataframe and populate the R array with ratings
        for i, row in df.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']
        print(R)
        print("\n")

    elif(y==2):
        #Γράφημα με Ευκλείδεια και συνημιτονοειδή μετρική
        User = df['User'].unique()
        Movie= df['Movie'].unique()

        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie))) 

        # Loop through the dataframe and populate the R array with ratings
        for i, row in df.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']
        
        try:
            answer = int(input("Enter the number of Cluster(s) - L you want to get the Plot (Empty->Best Elbow method): "))
        except ValueError:
            answer = None
        if answer is None:
            answer=optimal_clusters('euclidean',R)
            print("Euclidean clusters Elbow=" + str(answer))
            make_plot('euclidean',answer,R)
            
            answer=optimal_clusters('cosine',R)
            print("Cosine clusters Elbow=" + str(answer))
            make_plot('cosine',answer,R)
        else:
           print("Euclidean clusters =" + str(answer))
           make_plot('euclidean',answer,R)
           print("Cosine clusters =" + str(answer))
           make_plot('cosine',answer,R)

    elif (y == 3):

        Rmin = 30
        Rmax = 40
        # Group by and filter dataframe based on number of ratings per user
        user_counts = df.groupby('User').count()['Movie']
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        df_filtered = df[df['User'].isin(user_counts_filtered.index)]

        # Print number of users with ratings between Rmin and Rmax
        num_users_filtered = len(user_counts_filtered)
        print(f"Number of users with ratings between {Rmin} and {Rmax}: {num_users_filtered}")
        print(user_counts_filtered)

        # Create a list of unique users and movies
        User = df_filtered['User'].unique()
        Movie = df_filtered['Movie'].unique()

        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie)))

        # Loop through the filtered dataframe and populate the R array with ratings
        for i, row in df_filtered.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']

        Jaccard_array = jaccard_distances(R)
        try:
            answer = int(input("Enter the number of Cluster(s) - L you want to get the Plot for jaccard distance (Empty->Best Elbow method): "))
        except ValueError:
            answer = None
        if answer is None:
            answer = optimal_clusters('euclidean', Jaccard_array)
            print("Euclidean clusters Elbow=" + str(answer))
            #make_plot('euclidean', answer, R)
            make_plot('euclidean', answer, Jaccard_array)
            answer = optimal_clusters('cosine', Jaccard_array)
            print("Cosine clusters Elbow=" + str(answer))
            #make_plot('cosine', answer, R)
            make_plot('cosine', answer, Jaccard_array)
        else:

            # print("Jaccard clusters Elbow=" + str(answer))
            #make_plot('jaccard', answer, R)
            print("Euclidean clusters=" + str(answer))
            #make_plot('euclidean', answer, R)
            make_plot('euclidean', answer, Jaccard_array)
            print("Cosine clusters =" + str(answer))
            #make_plot('cosine', answer, R)
            make_plot('cosine', answer, Jaccard_array)

        # Determine the optimal number of clusters using the elbow method for the Euclidean distance measure
        num_clusters = answer

        # Cluster the data using k-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(Jaccard_array)

        # Get the labels assigned to each data point
        labels = kmeans.labels_

        # Create an empty dictionary to hold the ratings for each cluster
        cluster_users = {}

        # Loop through each cluster and extract the users in that cluster
        for i in range(num_clusters):
            # Find the indices of the data points in this cluster
            indices = np.where(labels == i)[0]

            # Extract the users for those data points
            cluster_User = User[indices]

            # Add the users to the dictionary
            cluster_users[i] = cluster_User

        # Print the users for each cluster
        for i in range(num_clusters):
            print(f"Users in cluster {i}:")
            print(cluster_users[i])
            input_user = int(np.random.choice(cluster_users[i]))

    elif (y == 5):

        Rmin = 30
        Rmax = 40
        # Group by and filter dataframe based on number of ratings per user
        user_counts = df.groupby('User').count()['Movie']
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        df_filtered = df[df['User'].isin(user_counts_filtered.index)]

        # Print number of users with ratings between Rmin and Rmax
        num_users_filtered = len(user_counts_filtered)
        print(f"Number of users with ratings between {Rmin} and {Rmax}: {num_users_filtered}")
        print(user_counts_filtered)

        # Create a list of unique users and movies
        User = df_filtered['User'].unique()
        Movie = df_filtered['Movie'].unique()

        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie)))

        # Loop through the filtered dataframe and populate the R array with ratings
        for i, row in df_filtered.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']

        Jaccard_array = jaccard_distances(R)
        try:
            answer = int(input("Enter the number of Cluster(s) - L you want to get the Plot for jaccard distance (Empty->Best Elbow method): "))
        except ValueError:
            answer = None
        if answer is None:

            answer = optimal_clusters('euclidean', Jaccard_array)
            print("Euclidean clusters Elbow=" + str(answer))
            #make_plot('euclidean', answer, R)
            make_plot('euclidean', answer, Jaccard_array)
            answer = optimal_clusters('cosine', Jaccard_array)
            print("Cosine clusters Elbow=" + str(answer))
            #make_plot('cosine', answer, R)
            make_plot('cosine', answer, Jaccard_array)
        else:

            # print("Jaccard clusters Elbow=" + str(answer))
            #make_plot('jaccard', answer, R)
            print("Euclidean clusters=" + str(answer))
            #make_plot('euclidean', answer, R)
            make_plot('euclidean', answer, Jaccard_array)
            print("Cosine clusters =" + str(answer))
            #make_plot('cosine', answer, R)
            make_plot('cosine', answer, Jaccard_array)

        # Determine the optimal number of clusters using the elbow method for the Euclidean distance measure
        num_clusters = answer

        # Cluster the data using k-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(Jaccard_array)

        # Get the labels assigned to each data point
        labels = kmeans.labels_

        # Create an empty dictionary to hold the ratings for each cluster
        cluster_users = {}

        # Loop through each cluster and extract the users in that cluster
        for i in range(num_clusters):
            # Find the indices of the data points in this cluster
            indices = np.where(labels == i)[0]

            # Extract the users for those data points
            cluster_User = User[indices]

            # Add the users to the dictionary
            cluster_users[i] = cluster_User

        # Print the users for each cluster
        for i in range(num_clusters):
            print(f"Users in cluster {i}:")
            print(cluster_users[i])
            input_user = int(np.random.choice(cluster_users[i]))
            train_faction(input_user)

        print("\n--------------------\n")

    elif (y == 4):

        Rmin = 40
        Rmax = 50
        # Group by and filter dataframe based on number of ratings per user
        user_counts = df.groupby('User').count()['Movie']
        user_counts_filtered = user_counts[user_counts.between(Rmin, Rmax)]
        df_filtered = df[df['User'].isin(user_counts_filtered.index)]

        # Print number of users with ratings between Rmin and Rmax
        num_users_filtered = len(user_counts_filtered)
        print(f"Number of users with ratings between {Rmin} and {Rmax}: {num_users_filtered}")
        #print(user_counts_filtered)

        # Create a list of unique users and movies
        User = df_filtered['User'].unique()
        Movie = df_filtered['Movie'].unique()

        # Create a 2D numpy array filled with zeros
        R = np.zeros((len(User), len(Movie)))

        # Loop through the filtered dataframe and populate the R array with ratings
        for i, row in df_filtered.iterrows():
            user_index = np.where(User == row['User'])[0][0]
            movie_index = np.where(Movie == row['Movie'])[0][0]
            R[user_index, movie_index] = row['Rate']

        # Define the number of nearest k-1 neighbors to consider
        k = 11

        # Compute the Jaccard similarity matrix for all users in the filtered dataset
        simularity_matrix = 1 - jaccard_distances(R)

        # Create a list to store the top k users for each user in the filtered dataset
        neighbor_list = []

        print(simularity_matrix)
        print(jaccard_distances(R))
        print('/n')

        id_answer_list=[]
        id_dictionary={user: {} for user in User}
        # Loop over each user in the filtered dataset
        for i, user in enumerate(User):
            # Get the row of the Jaccard similarity matrix corresponding to the current user
            jaccard_row = simularity_matrix[i, :]

            # Sort the Jaccard similarity row in descending order and get the indices of the top k users
            topk_indices = np.argsort(-jaccard_row)[:k]

            # Get the user IDs of the top k users (excluding the current user)
            topk_users = [User[idx] for idx in topk_indices if User[idx] != user]

            # Store the top k users in the neighbor list for the current user
            neighbor_list.append(topk_users)

            # Print the top k users for the current user
            #print(f"Neighbors of user {user}: {topk_users}")

            id_answer_list.append(user)
            id_dictionary[user]={'user' : user,'neighbor_list' : neighbor_list}
            neighbor_list=[]

        # Loop over each user in the answer list
        for user in id_answer_list:
            # Print the neighbor list for the current user
            print(f"Neighbor list for user {user}: {id_dictionary[user]['neighbor_list']}")

        print(id_answer_list)

        ratings_answers = []
        for user in id_answer_list:
            # Find the index of the user in the User array
            user_index = np.where(User == user)[0][0]

            # Access the row of the R array corresponding to the user's index to get the ratings for all movies
            user_ratings = R[user_index, :]
            ratings_answers.append(user_ratings)

        print(ratings_answers)
        print(len(ratings_answers))

        ratings_dictionary = {user: {} for user in User}
        for user_id, user_data in id_dictionary.items():
            user_ratings = []
            for neighbor_list in user_data['neighbor_list']:
                for neighbor in neighbor_list:
                    neighbor_index = np.where(User == neighbor)[0][0]
                    neighbor_ratings = R[neighbor_index, :]
                    ratings_dictionary[user_id][neighbor] = neighbor_ratings

        for user_id, user_data in ratings_dictionary.items():
            neighbor_ratings = user_data.values()
            print(f"Ratings of neighbors for user {user_id}: {list(neighbor_ratings)}")

        manual_input = input("Do you want to input a user manually? (y/n)")

        if manual_input.lower() == "y":
            # Prompt the user to input the user ID
            #input_user = int(input("Enter the ID of the user:"))
            input_user = 21278766
        else:
            # Select a random user from the dataframe
            input_user = int(df_filtered['User'].sample().iloc[0])

            # Print the selected user
            print("Randomly selected user:", input_user)

        # Check if input user exists in dataset
        if input_user not in User:
            print("User not found in dataset.")
        else:
            # Find the index of the user in the User array
            user_index = np.where(User == input_user)[0][0]

            # Access the row of the R array corresponding to the user's index to get the ratings for all movies
            user_ratings = R[user_index, :]

            # Print the ratings of the input user for all movies
            print(f"Ratings of user {input_user} for all movies: {user_ratings}")

            ratings_dictionary = {user: {} for user in User}

            for user_id, user_data in id_dictionary.items():
                user_ratings = []
                for neighbor_list in user_data['neighbor_list']:
                    for neighbor in neighbor_list:
                        neighbor_index = np.where(User == neighbor)[0][0]
                        neighbor_ratings = R[neighbor_index, :]
                        ratings_dictionary[user_id][neighbor] = neighbor_ratings

            train_ratings_dict = {}
            test_ratings_dict = {}

            # Loop over each user and their neighbor ratings
            for user_id, user_data in ratings_dictionary.items():
                # Collect the neighbor ratings into a list
                neighbor_ratings_list = list(user_data.values())

                # Combine the neighbor ratings into a 2D numpy array
                input_features = np.array(neighbor_ratings_list)

                # Create an array of actual ratings for the input user, based on the ratings_answers dictionary
                actual_rating = np.array([ratings_answers[np.where(User == user_id)[0][0]]] * len(input_features))

                # Split the input features and actual ratings into training and testing sets
                train_features, test_features, train_labels, test_labels = train_test_split(
                    input_features, actual_rating, test_size=0.2, random_state=42
                )

                # Create a dictionary containing the training features and labels for the current user
                train_ratings_dict[user_id] = {'input_features': train_features, 'labels': train_labels}

                # Create a dictionary containing the testing features and labels for the current user
                test_ratings_dict[user_id] = {'input_features': test_features, 'labels': test_labels}

            # Combine the training features and labels for all users into a single dataframe and numpy array, respectively
            x_train = pd.concat(
                [pd.DataFrame(train_ratings_dict[user_id]['input_features']) for user_id in train_ratings_dict.keys()],
                ignore_index=True)
            y_train = np.concatenate([train_ratings_dict[user_id]['labels'] for user_id in train_ratings_dict.keys()])

            # Combine the testing features and labels for all users into a single dataframe and numpy array, respectively
            x_test = pd.concat(
                [pd.DataFrame(test_ratings_dict[user_id]['input_features']) for user_id in test_ratings_dict.keys()],
                ignore_index=True)
            y_test = np.concatenate([test_ratings_dict[user_id]['labels'] for user_id in test_ratings_dict.keys()])

            model = Sequential()
            model.add(Embedding(len(Movie) + 1, 16, input_length=x_train.shape[1]))
            model.add(Flatten())
            model.add(Dense(len(Movie), activation='linear'))

            #model.add(Dense(64, activation='relu'))
            #model.add(Dense(32, activation='relu'))
            #model.add(Dense(16, activation='linear'))
            #model.add(Dense(32, activation='relu'))
            #model.add(Dense(len(Movie), activation='relu'))
            model.add(Dense(len(Movie), activation='relu'))
            model.add(Dense(len(Movie), activation='sigmoid'))
            #model.add(Dense(len(Movie), activation='softmax'))
            #model.add(Dense(len(Movie), activation='relu'))

            #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'mean_squared_error'])
            #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', 'mean_squared_error'])
            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy','mean_squared_error'])
            # model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['accuracy','mean_squared_error'])
            model.fit(x_train, y_train, epochs=200, batch_size=32)

            print(len(x_test))
            print(len(x_train))
            print(len(y_test))
            print(len(y_train))

            #print(x_test)
            #print(x_train)
            #print(y_test)
            #print(y_train)

            y_hat = model.predict(x_test)
            print(y_hat)
            print(y_test)

os.system('cls' if os.name == 'nt' else 'clear')
loading_data()
solution()
