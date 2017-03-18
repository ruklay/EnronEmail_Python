import pyodbc
import os.path
import numpy as np
import difflib
import datetime
import math

Mode = "TFIDF" #TDI TFIDF

def get_thread_index(min_tid, thread_no):
    return thread_no - min_tid

def get_thread_no(min_tid, index):
    return min_tid + index

def get_doc_index(doc_list, doc_name):
    if(doc_name in doc_list):
        doc_index = doc_list.index(doc_name)+1
    else:
        doc_index = -1
    return doc_index

def cluster(method, dis_matrix, n, Mode, ef, p):
    if(method == "KMeans"):
        M = "KM"
        #print("KMeans n = "+str(n))
        from sklearn.cluster import KMeans
        #eigen_values, eigen_vectors = np.linalg.eigh(dis_matrix)
        clusters_KMeans = KMeans(n_clusters=n, init='k-means++').fit(dis_matrix)
        labels = clusters_KMeans.labels_
        output_file = open("C:/tmp2/result_cluster/"+Mode+"/"+str(p)+"_out_cluster_KMeans_"+str(n)+".txt", "w")
        for item in labels:
            output_file.write("%s\n" % item)
        output_file.close()
        
    elif (method == "Spectral"):
        M = "SP"
        #print("SpectralClustering n = "+str(n))
        from sklearn.cluster import SpectralClustering
        cl = SpectralClustering(n_clusters=n,affinity='precomputed')
        clusters_SpectralClustering = cl.fit(dis_matrix)
        labels = clusters_SpectralClustering.labels_
        output_file = open("C:/tmp2/result_cluster/"+Mode+"/"+str(p)+"_out_cluster_SpectralClustering_"+str(n)+".txt", "w")
        for item in labels:
            output_file.write("%s\n" % item)
        output_file.close()
        
    else:
        print("No method found")
        return -1
    

    from sklearn import metrics
    ef.write("%d%%\t%s\tn=%d\t%.7f\t%.7f\n" % (p, M, n, metrics.silhouette_score(dis_matrix, labels, metric='euclidean'), metrics.calinski_harabaz_score(dis_matrix, labels)))
    #ef.write("%d0%% %s n=%d C-Score: %.7f\n" % (p, M, n, metrics.calinski_harabaz_score(dis_matrix, labels)))

    
    return 0
    
    #print("%d0%% %s n=%d S Score: %f" % (1, M, n, -0.01))

#--------------------------------------------------------
# Get data from the database

print("Running on Mode: "+Mode)
print("Start getting data from DB: "+str(datetime.datetime.now()))

cnxn = pyodbc.connect('Trusted_Connection=yes', driver = '{ODBC Driver 13 for SQL Server}',server = 'localhost', database = 'Enron')
cursor = cnxn.cursor()

min_msg_no = 30
max_msg_no = 31

SqlCommand = "select min(tid) as min_tid, max(tid) as max_tid from thread2 where messageno in (" + str(min_msg_no) + "," + str(max_msg_no) + ")"
cursor.execute(SqlCommand)
rows = cursor.fetchall()

min_tid = rows[0].min_tid
max_tid = rows[0].max_tid
num_thread = max_tid - min_tid + 1
#print("min_tid = "+str(min_tid))
#print("max_tid = "+str(max_tid))
print("num_thread = "+str(num_thread))

SqlCommand = "select tid, mid from dbo.message3 where tid >=" + str(min_tid) + "and tid <=" + str(max_tid) + "order by tid, date"
cursor.execute(SqlCommand)
rows = cursor.fetchall()
res = np.array(rows)
#print(np.shape(res))

SqlCommand = "select name from dbo.document3"
cursor.execute(SqlCommand)
rows = cursor.fetchall()

doc_list = []
for row in rows:
    doc_list.append(row.name)

cnxn.close()

#--------------------------------------------------------
num_word_all = 7593
ef = open("C:/tmp2/result_cluster/"+Mode+"/00_evaluation2.txt", "w")

for p in range(1,10):
    
    print("Top "+str(p)+" % "+Mode)
    
    #--------- Get Top Percent Term ID
    N = math.ceil(p*num_word_all/100)
    file_name_Top = "C:/tmp/Top"+Mode+".txt"
    with open(file_name_Top, 'r') as f:
       Top_list = [int(line.rstrip('\n')) for line in f][:N]
    #print(len(Top_list))
    
    #--------- Prepare lists of concattinated document id of each thread
    print("   Preparint lists: "+str(datetime.datetime.now()))
    list_tids = [[]] * num_thread
                
    for i in range(min_tid, max_tid+1):
        list_tid = []
        #print("Performing tid: "+str(i)+" of "+str(max_tid))
        
        sub_array = res[res[:,0]==i][:,1]
        for j in range(0,len(sub_array)):
            #print(sub_array[i])
            file_name = "C:/tmp/AR/body_noun_mid_3_to_51_filter/" + str(sub_array[j]) + ".txt"
            #print(file_name)
            if(os.path.isfile(file_name)):
                with open(file_name, 'r') as f:
                    tmp_list = [get_doc_index(doc_list, line.rstrip('\n')) for line in f]
            else:
                tmp_list = []
            #print(str(i)+str(tmp_list))
            list_tid = list_tid + tmp_list
    
        list_tids[get_thread_index(min_tid,i)] = [i for i in list_tid if i in Top_list]
    
    #--------- Calculate distance matrix
    print("   Calculate distance matrix: "+str(datetime.datetime.now()))
    dis_matrix = np.zeros(shape=(num_thread,num_thread))

    for i in range(0, num_thread):
        for j in range(i, num_thread):
            if i!=j:
                sm = difflib.SequenceMatcher(None,list_tids[i],list_tids[j])
                dis = 1-sm.ratio()
                dis_matrix[i,j] = dis
                dis_matrix[j,i] = dis

    # ------- Perform clustering
    print("   Perform clustering: "+str(datetime.datetime.now()))
    
    cluster("KMeans", dis_matrix, 2, Mode, ef, p)
    cluster("KMeans", dis_matrix, 3, Mode, ef, p)
    cluster("KMeans", dis_matrix, 4, Mode, ef, p)
    cluster("KMeans", dis_matrix, 5, Mode, ef, p)
    cluster("KMeans", dis_matrix, 6, Mode, ef, p)
    
    cluster("Spectral", dis_matrix, 2, Mode, ef, p)
    cluster("Spectral", dis_matrix, 3, Mode, ef, p)
    cluster("Spectral", dis_matrix, 4, Mode, ef, p)
    cluster("Spectral", dis_matrix, 5, Mode, ef, p)
    cluster("Spectral", dis_matrix, 6, Mode, ef, p)
    
    print("")
    
ef.close()















