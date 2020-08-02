import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
#from sklearn.decomposition import PCA

df=pd.read_csv("CarPrice_Assignment.csv")

#补充大类品牌
df['Brand']=df['CarName'].str.split(' ',expand=True,n=1)[0]
df['Sub_Brand']=df['CarName'].str.split(' ',expand=True,n=1)[1]

#修正品牌
df.Brand[df['Brand']=='vw']='volkswagen'
df.Brand[df['Brand']=='maxda']='mazda'
df.Brand[df['Brand']=='Nissan']='nissan'
df.Brand[df['Brand']=='porcshce']='porsche'
df.Brand[df['Brand']=='toyouta']='toyota'
df.Brand[df['Brand']=='vokswagen']='volkswagen'

#数据探索
#df.profile_report()

#变量选择
input_model = df.drop(['car_ID','CarName','Brand','Sub_Brand'], axis=1)

#分类变量因子化
le = LabelEncoder()
input_model['fueltype'] = le.fit_transform(input_model['fueltype'])
input_model['aspiration'] = le.fit_transform(input_model['aspiration'])
input_model['doornumber'] = le.fit_transform(input_model['doornumber'])
input_model['carbody'] = le.fit_transform(input_model['carbody'])
input_model['drivewheel'] = le.fit_transform(input_model['drivewheel'])
input_model['enginelocation'] = le.fit_transform(input_model['enginelocation'])
input_model['enginetype'] = le.fit_transform(input_model['enginetype'])
input_model['cylindernumber'] = le.fit_transform(input_model['cylindernumber'])
input_model['fuelsystem'] = le.fit_transform(input_model['fuelsystem'])

#标准化
min_max_scaler = preprocessing.MinMaxScaler()
input_model = min_max_scaler.fit_transform(input_model)

pd.DataFrame(input_model).to_csv("input_model.csv")

#手肘法
sse = []
for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(input_model)
    sse.append(kmeans.inertia_)
x = range(2,10)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x,sse,'o-')
plt.show()

#轮廓系数
sc_scores = []
for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans_model = kmeans.fit(input_model)
    sc_score = silhouette_score(input_model,kmeans_model.labels_,metric='euclidean')
    sc_scores.append(sc_score)
k=[2,3,4,5,6,7,8,9]
plt.xlabel('k')
plt.ylabel('SCS')
plt.plot(k,sc_scores,'*-')
plt.show()

#k=5
kmeans = KMeans(n_clusters=5)
kmeans.fit(input_model)
pre = kmeans.predict(input_model)

df['pre'] = pre
df.to_csv('project_c_result.csv', encoding='utf-8')

#输出volkswagen所在组数
vw_list = df.loc[df['Brand'].str.contains('volkswagen')]
vw_group = vw_list['pre'].to_list()
competitor_list=[]

#输出同在一组品牌
for i in vw_group:
	competitor_list += list(df[df['pre'] == i]['Brand'].unique())

#去重
competitor_list = list(set(competitor_list))
competitor_list.remove('volkswagen')

#输出结果
print(("经聚类，竞品包括：{}").format(competitor_list))