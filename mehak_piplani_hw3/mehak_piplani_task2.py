import json
import networkx as nx
import sys
import time

#read in the json file
tweets = []
input_path = sys.argv[1]
output_part_a= sys.argv[2]
output_part_b = sys.argv[3] 
output_part_c = sys.argv[4] 
dict_user_to_number={}
data={}
i=0
with open(input_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        J = json.loads(line)
        tweets.append(J)
start_time = time.time()
G = nx.Graph() 
      
for tweet in tweets:
    author = tweet['user']['screen_name']
    if author not in G:
        G.add_node(author)
    if author not in dict_user_to_number:
        
        dict_user_to_number[author]=i
        data[i]=tweet['text']
        i+=1
    else:
        data[dict_user_to_number[author]]+=' '+tweet['text']
            
    try:
        rtauthor = tweet['retweeted_status']['user']['screen_name']
    
        
        if rtauthor != "" and not rtauthor in G:
            G.add_node(rtauthor)
        if rtauthor!="":
            
        
            
            if rtauthor not in dict_user_to_number:
           
                dict_user_to_number[rtauthor]=i
                data[i]=tweet['retweeted_status']['text']
                i+=1
            
            else:
                data[dict_user_to_number[rtauthor]]+=' ' + tweet['retweeted_status']['text']
            
            if G.has_edge(rtauthor, author) or G.has_edge(author,rtauthor):
                G[rtauthor][author]['weight'] += 1
            
            else:
                G.add_edge(rtauthor,author, weight=1.0)
    except:
        continue
            


def edge_to_remove(graph):
  G_dict = nx.edge_betweenness_centrality(graph,normalized=False,weight="weight")
  sorted_g_dict=sorted(G_dict.items(), key=lambda item: item[1], reverse = True)
  
  edge,max_value=sorted_g_dict[0]
  
  edges = [edge]
  for key, value in sorted_g_dict:
    
    if key!=edge and value==max_value:
      edges.append(key)
    if value<max_value:
        break
 
  return edges
  
  
def CalModularity(graph,deg_,m):
    
    Mod=0
    for c in nx.connected_components(graph):
        c=list(c)
        for i in range(len(c)-1):
            for j in range(i,len(c)):
                if c[i]!=c[j]:
                    pair = tuple(sorted([c[i], c[j]]))
                    try:
                        a_i_j = graph.get_edge_data(pair[0],pair[1])["weight"]
                        #print(a_i_j)
                    except:
                        a_i_j=0.0
                    
                    Mod += (a_i_j - ((deg_[c[i]] * deg_[c[j]]) / (2.0 * m)))
                   
    Mod = Mod/float(2*m)
   
    return Mod

    
def girvan_newman(graph,max_modularity,Orig_deg,m):
    
    sg_count = nx.number_connected_components(graph)
    while True:

        if graph.number_of_edges() == 0:
            break
        remove=edge_to_remove(graph)
        for i in remove:
            
            graph.remove_edge(i[0], i[1])
        
        sg_new =  nx.number_connected_components(graph)
        
        if sg_new > sg_count:
            New_modularity=CalModularity(graph,Orig_deg,m)
            #print("check",New_modularity)
           
            if New_modularity>max_modularity:
                max_modularity=New_modularity
                max_Communities=list(nx.connected_components(graph))
                #print(max_modularity,list(max_Communities))
                print(nx.number_connected_components(graph))

        sg_count=sg_new
    return max_Communities,max_modularity
    



max_modularity=-float("inf")
A = nx.adj_matrix(G)    #adjacenct matrix

m=0
for i in G.edges(data=True):
    m+=i[2]["weight"]
print(m)

Orig_deg = {}
for i in G.degree(weight='weight'):
    Orig_deg[i[0]]=i[1]
c,max_modularity = girvan_newman(G,max_modularity,Orig_deg,m)


node_groups = []
#print(max_modularity)

outfile = open(output_part_a, "w+")
outstring=""
outstring = "Best Modularity is: "+str(max_modularity)
outstring+="\n"
count=0
main=[]
for i in c:
    temp=[]
    temp=list(i)
    main.append(sorted(temp))
    

for i in sorted(main,key = lambda x: (len(x),x)):
    elements = ["'"+str(element)+"'" for element in i]
    if len(i)>=2:
        string = ",".join(elements)
        outstring+=string
    else:
      outstring+=str(elements[0])
    outstring+="\n"
    count+=1
    if count == len(main)-1:
        community_1=i
    if count == len(main):
        community_2=i
        
outfile.write(outstring)
outfile.close()
train_data=[]
train_data_1=[]
test_data=[]
train_label=[]
train_label_1=[]
test=[]


for i in dict_user_to_number.keys():
    if i in community_1:
        
        train_data.append(data[dict_user_to_number[i]])
        train_label.append(0)
    elif i in community_2:
        train_data_1.append(data[dict_user_to_number[i]])
        train_label_1.append(1)
    else:
        test_data.append(data[dict_user_to_number[i]])
        test.append(i)

train_data=train_data+ train_data_1

train_label=train_label+train_label_1


from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_counts = vectorizer.fit_transform(train_data)

dict_number_to_user = {v: k for k, v in dict_user_to_number.items()}

#train a multinomial naive Bayes classifier
clf = MultinomialNB().fit(X_train_counts, train_label)
docs_test = vectorizer.transform(test_data)

predicted = clf.predict(docs_test)

community_0_pred=[]
community_1_pred=[]
for i in range(len(predicted)):
    if predicted[i]==0:
        community_0_pred.append(test[i])
    else:
        community_1_pred.append(test[i])
communtiy_0_pred=community_0_pred.extend(community_1)  
communtiy_1_pred=community_1_pred.extend(community_2)
print(len(community_0_pred))
print(len(community_1_pred))
outfile = open(output_part_b, "w+")
outstring=""

 
if len(community_0_pred)<len(community_1_pred):
    elements = ["'"+str(element)+"'" for element in sorted(community_0_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    elements = ["'"+str(element)+"'" for element in sorted(community_1_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
else:
    
    elements = ["'"+str(element)+"'" for element in sorted(community_1_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    elements = ["'"+str(element)+"'" for element in sorted(community_0_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    
outfile.write(outstring)
outfile.close()


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train_data)

dict_number_to_user = {v: k for k, v in dict_user_to_number.items()}

#train a multinomial naive Bayes classifier
clf = MultinomialNB().fit(X_train_counts, train_label)
docs_test = vectorizer.transform(test_data)

predicted = clf.predict(docs_test)

community_0_pred=[]
community_1_pred=[]
for i in range(len(predicted)):
    if predicted[i]==0:
        community_0_pred.append(test[i])
    else:
        community_1_pred.append(test[i])
communtiy_0_pred=community_0_pred.extend(community_1)    
communtiy_1_pred=community_1_pred.extend(community_2)  

outfile = open(output_part_c, "w+")
outstring=""

 
if len(community_0_pred)<len(community_1_pred):
 
    elements = ["'"+str(element)+"'" for element in sorted(community_0_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    elements = ["'"+str(element)+"'" for element in sorted(community_1_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
else:
    
    elements = ["'"+str(element)+"'" for element in sorted(community_1_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    elements = ["'"+str(element)+"'" for element in sorted(community_0_pred)]

    string = ",".join(elements)
    outstring+=string
    outstring+="\n"
    
outfile.write(outstring)
outfile.close()