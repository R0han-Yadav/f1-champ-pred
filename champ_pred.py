# importing libs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import folium
from folium import plugins
import webbrowser
from IPython.display import display, HTML
import warnings
warnings.simplefilter("ignore")

# csv to dataframe
result_df = pd.read_csv('results.csv')
stats_df = pd.read_csv('status.csv')
drivers_df = pd.read_csv('drivers.csv')
races_df = pd.read_csv('races.csv')
constructor_df = pd.read_csv('constructors.csv')
driver_standings_df = pd.read_csv('driver_standings.csv')
pd.get_option("display.max_columns", None)

# printing dataframe head
result_df.head()
stats_df.head()
drivers_df.head()
races_df.head()
constructor_df.head()
driver_standings_df.head()

# merging dataframes
con1 = pd.merge(result_df, races_df, on='raceId')
con2 = pd.merge(con1, drivers_df, on='driverId')
con3 = pd.merge(con2, driver_standings_df, on='driverId')
con4 = pd.merge(con3, constructor_df, on='constructorId')
df = pd.merge(con4, stats_df, on='statusId')
pd.get_option("display.max_columns", None)

# printing merged dataframe
df.head()

df.isna().sum()

df.info()

df.describe().T

df.columns

# simplifying dataframe
df = df.drop(['url','url_x','position_x','fastestLapTime','positionText_x','time_x','time_y','driverRef',
              'constructorRef','nationality_y','url_y','positionText_y','raceId_y','points_y'],1)

# renaming cols
col_name = {'number_x':'number','milliseconds':'timetaken_in_millisec','fastestLapSpeed':'max_speed',
 'name_x':'grand_prix','number_y':'driver_num','code':'driver_code','nationality_x':'nationality','name_y':'company',
 'raceId_x':'racerId','points_x':'points','position_y':'position'}

df.rename(columns=col_name,inplace=True)

# printing renamed col dataframe
df.head()

# merging forename and surname for full name(driver_name)
df['driver_name'] = df['forename']+' '+df['surname']
df = df.drop(['forename','surname'],1)

# converting/finding dob
pd.to_datetime(df.date)
df['dob'] = pd.to_datetime(df['dob'])
df['date'] = pd.to_datetime(df['date'])

dates = datetime.today()-df['dob']
age = dates.dt.days/365

# age round off
df['age'] = round(age)
pd.set_option('display.max_columns', None)

# printing dataframe
df.head()

# removing errors
l = ['number','timetaken_in_millisec','fastestLap','rank','max_speed','driver_num']
for i in l:
    df[i] = pd.to_numeric(df[i],errors='coerce')

df.drop('driver_num',1,inplace=True)

cat = []
num = []
for i in df.columns:
    if df[i].dtypes == 'O':
        cat.append(i)
    else:
        num.append(i)
df[cat].head()

df[num].head()

df.head()

df.isnull().sum() / len(df) * 100

df['max_speed'].mean()

df[['rank', 'fastestLap']] = df[['rank', 'fastestLap']].fillna(0)
df['timetaken_in_millisec'] = df['timetaken_in_millisec'].fillna(df['timetaken_in_millisec'].mean())
df['max_speed'] = df['max_speed'].fillna(df['max_speed'].mean())
df['number'] = df['number'].fillna(0)

df.isnull().sum() / len(df) * 100

circuit_df = pd.read_csv('circuits.csv')
circuit_df.head()

# map to display circuit location from circuits.csv
coordinates = []
for lat, lng in zip(circuit_df['lat'], circuit_df['lng']):
    coordinates.append([lat, lng])

maps = folium.Map(zoom_start=50, tiles='Stamen Watercolor', control_scale=True)
for i, j, k in zip(coordinates, circuit_df.name, circuit_df.url):
    marker = folium.Marker(
        location=i,
        icon=folium.Icon(icon="star",color='cadetblue'),
        tooltip="<strong>{0}</strong>".format(j),
        popup="<a href={0}>moar info</a>".format(k))
    marker.add_to(maps)
display(maps)
maps.save('map.html')

df_fin = df[df['status'] == 'Finished']
df_fin.tail()

mean = df.max_speed.mean()
mean2 = df.fastestLap.mean()
df = df_fin[df_fin['max_speed']>mean]
df.head()

df = df[df['fastestLap']>mean2]

df.year.unique()
df = df[(df['age']<df['age'].mean()) & (df['year']>2012)]
df.head()

df.drop('date',1,inplace=True)
df.drop('dob',1,inplace=True)
df.drop('statusId',1,inplace=True)

df.skew()

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]
df.head()

plt.figure(figsize=(17,12))
sns.heatmap(df.corr(),annot=True)
plt.show()

num.remove('date')
num.remove('dob')
num.remove('statusId')

# density plot to check normalization
plt.figure(figsize=(15,50))
for i,j in zip(num,range(1,len(num)+1)):
    plt.subplot(11,2,j)
    sns.kdeplot(df[i],shade=True,color='darkblue')
    plt.show()


df.skew()

le = LabelEncoder()

for i in cat:
    df[i] = le.fit_transform(df[i])
df.head()

x = df.drop('driver_name',1)
y = df.driver_name

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=42)

clf = DecisionTreeClassifier(max_depth=5,random_state=1234)
clf.fit(xtrain, ytrain)

tree.export_text(clf)

fn = list(df.columns)
fn.remove('driver_name')

fig = plt.figure(figsize=(10, 10))
_ = tree.plot_tree(clf,
               feature_names=fn,
               filled=True)

lr = LogisticRegression(solver='sag')
dt = DecisionTreeClassifier()
rn = RandomForestClassifier()
knn = KNeighborsClassifier()
gb = GaussianNB()
sgd = SGDClassifier()

li = [lr, sgd, knn, gb, rn, dt]
d = {}
for i in li:
    i.fit(xtrain,ytrain)
    ypred = i.predict(xtest)
    print(i, ":", accuracy_score(ypred, ytest)*100)
    d.update({str(i): i.score(xtest, ytest)*100})

plt.figure(figsize=(15, 6))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(), d.values(), marker='o', color='plum', linewidth=4, markersize=13,
         markerfacecolor='gold', markeredgecolor='slategray')
plt.show()

norm = MinMaxScaler().fit(xtrain)

X_train_norm = np.transform(xtrain)

X_test_norm = np.transform(xtest)

li = [lr,sgd,rn,knn,gb,dt]
di = {}
for i in li:
    i.fit(X_train_norm,ytrain)
    ypred = i.predict(X_test_norm)
    print(i,":",accuracy_score(ypred,ytest)*100)
    di.update({str(i):i.score(X_test_norm,ytest)*100})

plt.figure(figsize=(15, 6))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(di.keys(),di.values(),marker='o',color='skyblue',linewidth=4,markersize=13,
         markerfacecolor='gold',markeredgecolor='black')
plt.show()


scaler = RobustScaler().fit(xtrain)

xtrain_scaled = scaler.transform(xtrain)

xtest_scaled = scaler.transform(xtest)
li = [lr,sgd,rn,knn,gb,dt]
dics = {}
for i in li:
    i.fit(xtrain_scaled,ytrain)
    ypred = i.predict(xtest_scaled)
    print(i,":",accuracy_score(ypred,ytest)*100)
    dics.update({str(i):i.score(xtest_scaled,ytest)*100})

plt.figure(figsize=(15, 6))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(dics.keys(),dics.values(),marker='o',color='darkseagreen',linewidth=4,markersize=13,
         markerfacecolor='gold',markeredgecolor='black')
plt.show()
