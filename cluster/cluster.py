import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from color_name import cnames

with open('end_point.json') as f:
    end_points=json.load(f)
    
end_points = np.asarray(end_points)
x1 = end_points[:, 0]
x2 = end_points[:, 1]

# fig = plt.figure(0, figsize=(8, 7), dpi=500)

# plt.xlim([-100, 100])
# plt.ylim([-100, 100])
# plt.title('Sample')

# plt.scatter(x1, x2, s=1)
# plt.savefig('./points.jpg')

clusters=64
kmeans_model = KMeans(n_clusters=clusters).fit(end_points)

# print('聚类结果：', kmeans_model.labels_)

colors = list(cnames.keys()) 
markers = ['o', 's', 'D']  
centers = kmeans_model.cluster_centers_.tolist()

with open('./cluster_centers.json', 'w') as f:
    json.dump(centers, f, indent=4)


fig = plt.figure(0, figsize=(8, 7), dpi=500)
plt.rcParams['axes.facecolor'] = 'silver'


plt.xlim([-100, 100])  
plt.ylim([-100, 100])  
plt.title('K = %s' %(clusters))
 
plt.plot(0, 0, color='g' , marker='D')

for i, l in enumerate(tqdm(kmeans_model.labels_)):   
    plt.scatter(x1[i], x2[i], s=1, color=colors[l])


for center in centers:
    plt.scatter(center[0], center[1], s=3, color='k')

plt.savefig('./clusters_1.jpg')
