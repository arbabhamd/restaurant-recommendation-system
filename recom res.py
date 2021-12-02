from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math
import random
import googlemaps
import webbrowser


def get_direction(loc):
    from datetime import datetime

    gmaps = googlemaps.Client(key='AIzaSyD9Ev5d5DlvqhItUoj84ehZkDci0Tb5U6o')

    # Geocoding an address
    geocode_result = gmaps.geocode(loc)
    print(geocode_result[0]['geometry']['location'])

    lat = geocode_result[0]['geometry']['location']['lat']
    lng = geocode_result[0]['geometry']['location']['lng']

    # Look up an address with reverse geocoding
    reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))
    print(reverse_geocode_result)
    # Request directions via public transit
    # now = datetime.now()
    # directions_result = gmaps.directions("Sydney Town Hall",
    #                                      "Parramatta, NSW",
    #                                      mode="transit",
    #                                      departure_time=now)
    # directions_result

    # 31.4815212,74.3030141
    print('going to chrome')
    webbrowser.open('https://www.google.com/maps/dir/31.4815212,74.3030141/' + str(lat) + ',' + str(
        lng) + '/@31.4999802,74.2292793,'
               '12z/data=!3m1!4b1!4m12!1m7!3m6!1s0x0:0x0!2zMzHCsDI4JzI5LjAiTiA3NMKwMjInNDMuOSJF!3b1!8m2!3d' + str(
        lat) + '!4d' + str(lng) + '!4m3!1m1!4e1!1m0')


# reading the dataset
data = pd.read_csv('zomato.csv')
dataset = data.copy()

p = None
ind = None


def prio(a):
    return abs((a[ind]) - p)


# deleting unnecessary data
del data['url']

data.isnull().mean() * 100

# Replace Bogus terms with NaN values
data['rate'] = data['rate'].replace('NEW', np.NaN)
data['rate'] = data['rate'].replace('-', np.NaN)
data = data.rename(
    columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# Convert str to float
X = data.copy()
X.head()
X.rate = X.rate.astype(str)
X.rate = X.rate.apply(lambda x: x.replace('/5', ''))
X.rate = X.rate.astype(float)

X.cost = X.cost.astype(str)
X.cost = X.cost.apply(lambda y: y.replace(',', ''))
X.cost = X.cost.astype(float)
le = preprocessing.LabelEncoder()
X['online_order'] = le.fit_transform(X['online_order'])
X['book_table'] = le.fit_transform(X['book_table'])
X.votes = X.votes.astype(float)

# Now all value related columns are float type.
# Replace missing values by deleting missing values
X_del = X.copy()
X_del.dropna(how='any', inplace=True)

# Remove duplicates
X_del.drop_duplicates(subset='name', keep='first', inplace=True)
address = X_del.iloc[:, [0, 1]]
del X_del['address']
location = 'hello'


def data_visualisation():
    global X_del
    # Pie chart for understanding persentage of people from each location
    s = set()
    d = dict()
    for i in X_del.city:
        if i not in s:
            s.add(i)
            d.update({i: 1})
        else:
            d[i] += 1

    labels = d.keys()
    sizes = d.values()
    colors = []

    for x in labels:
        rgb = (random.random(), random.random(), random.random())
        colors.append(rgb)

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Understanding percentage of people from each location\n\n")
    plt.axis('equal')
    plt.show()

    # Pie chart for understanding percentage of restaurants offering online delivery
    s = set()
    d = dict()

    for i in X_del.online_order:
        if i not in s:
            s.add(i)
            d.update({i: 1})
        else:
            d[i] += 1

    labels = d.keys()
    sizes = d.values()
    colors = []

    for x in labels:
        rgb = (random.random(), random.random(), random.random())
        colors.append(rgb)

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Understanding the type category from each location")
    plt.axis('equal')
    plt.show()

    # Bar chart showing avg cost of food for each area of the city.
    s = set()
    d = dict()

    for i in X_del.city:
        if i not in s:
            s.add(i)
            d.update({i: 1})
        else:
            d[i] += 1

    d_cost = dict()

    for i in s:
        d_cost.update({i: 0})

    for i in X_del.index:
        d_cost[X_del.city[i]] += X_del.cost[i]

    for i in s:
        d_cost[i] = d_cost[i] / d[i]

    plt.title("Average cost vs Location")
    plt.ylabel('Average cost')
    plt.bar(d.keys(), height=d_cost.values())
    plt.xticks(rotation=90)
    plt.show()

    # Scatter plot for votes vs cost
    x = X_del.cost
    y = X_del.votes

    plt.title('Scatter plot for cost vs votes')
    plt.scatter(x, y)
    plt.show()

    # Histogram for cost
    x = X_del.cost

    plt.title('Histogram plot for cost')
    plt.hist(x, bins=50)
    plt.show()

    # Bar chart showing avg no of cuisines offered in each area
    s = set()
    d = dict()

    count_d = dict()

    for i, j in zip(X_del.city, X_del.cuisines):
        if i not in s:
            s.add(i)
            count_d.update({i: 1})
            d.update({i: len(j.split(', '))})
        else:
            count_d[i] += 1
            d[i] += len(j.split(', '))

    for i, j in zip(d, count_d):
        d[i] = d[i] / count_d[i]

    plt.title("Average no of cuisines offered vs Location")
    plt.ylabel('Average no of cuisines')
    plt.bar(d.keys(), height=d.values())
    plt.xticks(rotation=90)
    plt.show()


class RestaurantRecommendationSystem:
    def __init__(self):
        print()
        print("------------Restaurant Recommendation System------------\n\n")
        print("To Skip the any queries Enter \"skip\":\n")
        self.takeInput()
        self.fit()
        self.predict()

    def takeInput(self):
        global X_del

        # Input Location
        self.input_location = input("Enter Location :")
        if (self.input_location != "skip"):
            X_del = X_del.loc[X_del['city'] == self.input_location]

        if (len(X_del) == 0):
            print("Invalid location Entered")
            sys.exit()

        # Input Restaurant Type
        self.input_rest_type = input("Enter Restuarant Type :")
        if (self.input_rest_type != "skip"):
            X_del = X_del.loc[X_del['rest_type'] == self.input_rest_type]

        if (len(X_del) == 0):
            print("Invalid Restuarant Type Entered")
            sys.exit()

        # Input Required Cost
        self.cos = input("Enter Cost :")
        if self.cos != "skip":
            self.cos = float(self.cos)
        else:
            self.cos = X_del['cost'].mean()
            print(self.cos, "Is the Value Selected")

        # Input Required Rating
        self.rat = input("Enter The Rating 1-5 :")
        if self.rat != "skip" and abs(float(self.rat)) <= 5:
            self.rat = abs(float(self.rat))
        else:
            self.rat = 5.0

        # Input the required priority
        self.prior = int(
            input("\nEnter 0 to give no priority , 1 to give priority to Cost and 2 to give priority to Rating: "))

        # The max number of recommendations that can be computed is found out by len(X)-1
        # Input the number of recommendations
        self.no_of_recomendations = int(input("Enter the no. of Recommendations :"))

    def fit(self):
        global X_del

        # Extracting the required columns for the model
        cluster_with_name = X_del.iloc[:, [10, 3, 0, 5]]

        # Adding the new cost,rating and user at the end of the dataset
        new_row = {'cost': self.cos, 'rate': self.rat, 'name': "user"}
        cluster_with_name = cluster_with_name.append(new_row, ignore_index=True)

        checkCluster = X_del.iloc[:, [10, 3]]

        new_row = {'cost': self.cos, 'rate': self.rat}
        checkCluster = checkCluster.append(new_row, ignore_index=True)

        # Visualizing Dendrogram
        dendrogram = sch.dendrogram(sch.linkage(checkCluster, method='ward'))
        # plt.title('Dendrogram')
        # plt.xlabel('Customers')
        # plt.ylabel('Euclidean distances')
        # plt.show()

        # Clustering if the number of restaurants in the dataset can be clustered
        if len(checkCluster) > 5:
            hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
            y_hc = hc.fit_predict(checkCluster)

            # self.X = cluster_with_name.values
            # plt.scatter(self.X[y_hc == 0, 0], self.X[y_hc == 0, 1],
            #             s=100, c='red', label='Cluster 1')
            # plt.scatter(self.X[y_hc == 1, 0], self.X[y_hc == 1, 1],
            #             s=100, c='blue', label='Cluster 2')
            # plt.scatter(self.X[y_hc == 2, 0], self.X[y_hc == 2, 1],
            #             s=100, c='green', label='Cluster 3')
            # plt.scatter(self.X[y_hc == 3, 0], self.X[y_hc == 3, 1],
            #             s=100, c='cyan', label='Cluster 4')
            # plt.scatter(self.X[y_hc == 4, 0], self.X[y_hc == 4, 1],
            #             s=100, c='magenta', label='Cluster 5')
            # plt.title('Clusters of Restuarants')
            # plt.xlabel('Cost $')
            # plt.ylabel('Rating(1-5)')
            # plt.legend()
            # plt.show()

            self.X = cluster_with_name.values
            selected_cluster_no = y_hc[-1]
            self.single_cluster_X = self.X[y_hc == selected_cluster_no,]  # numpy nd array
        else:
            print("Since the no of items in the filter dataset is less, the clustering is not done")
            self.X = cluster_with_name.values
            self.single_cluster_X = self.X

    def predict(self):

        global p
        global ind

        # Model exits if the number of recommendations is more than what is there in the dataset
        if self.no_of_recomendations >= len(self.X):
            print("These no. of recommendations cannot pe processed")
            sys.exit()

        # If the number of recommendations is lesser than what is present in the cluster
        # select the cluster for K nearest neighbours
        if self.no_of_recomendations < len(self.single_cluster_X):
            self.X = self.single_cluster_X

        # Scaling the cost and the rating columns
        sc = StandardScaler()
        self.X[:, -4:-2] = sc.fit_transform(self.X[:, -4:-2])

        # Applying K nearest neighbours to the scaled data
        nbrs = NearestNeighbors(n_neighbors=self.no_of_recomendations + 1, algorithm='auto').fit(self.X[:, -4:-2])
        distances, indices = nbrs.kneighbors(self.X[:, -4:-2])
        self.X[:, -4:-2] = sc.inverse_transform(self.X[:, -4:-2])

        # Relevant cost or rating chosen as per priority of the user for sorting
        if self.prior == 1:
            p = self.cos
            ind = 0
        if self.prior == 2:
            p = self.rat
            ind = 1

        # Extract all the data of the K nearest neighbours in a list except the users
        count = 0
        l = []
        for i in indices[-1]:
            if self.X[i][2] == "user":
                continue
            l.append(self.X[i].tolist())

        # Sorting as per priority
        if self.prior != 0:
            l = sorted(l, key=prio)

        # Displaying the restaurant as per the users requirement
        count = 0
        sum_cost = 0
        sum_rat = 0
        recommendation_list = []
        for i in l:
            count = count + 1
            restaurant_info = {
                "Name": i[2],
                "Cost": i[0],
                "Rating": i[1],
                "Address": address.loc[address['name'] == i[2]].address.iloc[0],
                "Phone Number": i[3].split()[0] + " " + i[3].split()[1],
            }
            print("\n", count, "| Name:", restaurant_info["Name"], "| Cost: ", restaurant_info["Cost"],
                  "| Rating: ", restaurant_info["Rating"], "| Address: ", restaurant_info["Address"],
                  "| Phone Number: ", restaurant_info["Phone Number"])
            recommendation_list.append(restaurant_info)
            sum_cost += pow(i[0] - self.cos, 2)
            sum_rat += pow(i[1] - self.rat, 2)

        print("Cost Error in Rupees: ", math.sqrt(sum_cost / count))
        print("Rating Error: ", math.sqrt(sum_rat / self.rat))
        print()
        choice = int(input("Enter serial number (1-{number}) of restaurant you want to visit: ".format(
            number=len(recommendation_list))))
        selected_restaurant = recommendation_list[choice - 1]
        print("You have selected:")
        print(choice, "| Name:", selected_restaurant["Name"], "| Cost: ", selected_restaurant["Cost"],
              "| Rating: ", selected_restaurant["Rating"], "| Address: ", selected_restaurant["Address"],
              "| Phone Number: ", selected_restaurant["Phone Number"], "\n")
        location = selected_restaurant["Address"]
        decision = int(input('enter 1 if you want directions to your selected restaurant  or   enter 2 to skip: '))
        if decision == 1:
            print('hello')
            get_direction(location)


print()
print()
ch = int(input("Enter 1 for DataVisualization, 2 for Recommendation System, Enter any other number for both: "))
print()
if ch == 1:
    data_visualisation()
elif ch == 2:
    rrs = RestaurantRecommendationSystem()
else:
    data_visualisation()
    print()
    rrs = RestaurantRecommendationSystem()
