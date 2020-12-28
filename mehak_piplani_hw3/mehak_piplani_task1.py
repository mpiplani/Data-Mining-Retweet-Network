import json
import networkx as nx
import sys
import time
input_path = sys.argv[1]
output_path_graph = sys.argv[2]
output_path_file = sys.argv[3] 
tweets = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        J = json.loads(line)
        tweets.append(J)
start_time = time.time()
G = nx.DiGraph() 
#print(len(tweets))       
for tweet in tweets:
    author = tweet['user']['screen_name']
    if author not in G:
        G.add_node(author)
    try:
        rtauthor = tweet['retweeted_status']['user']['screen_name']
        
    except:
        continue
    
        
    if rtauthor != "" and not rtauthor in G:
        G.add_node(rtauthor)
    if rtauthor!="":
        if G.has_edge(author, rtauthor):
            G[author][rtauthor]['weight'] += 1
        else:
            G.add_weighted_edges_from([(author,rtauthor, 1.0)])

#print(time.time()-start_time)           
#save the graph as a gxef

num_of_node =  G.number_of_nodes()

num_of_edge=G.number_of_edges()

nx.write_gexf(G, output_path_graph)


degree_sequence_1 =sorted(G.in_degree(weight='weight'), key=lambda x: x[1], reverse=True)[0]
  
degree_sequence_2 =sorted(G.out_degree(weight='weight'), key=lambda x: x[1], reverse=True)[0]
#print(degree_sequence_1)
 
output_dict = {"n_nodes":num_of_node,"n_edges":num_of_edge,"max_retweeted_user":degree_sequence_1[0],"max_retweeted_number":int(degree_sequence_1[1]),"max_retweeter_user":degree_sequence_2[0],"max_retweeter_number":int(degree_sequence_2[1])}


with open(output_path_file, 'w') as file:
    json.dump(output_dict, file)