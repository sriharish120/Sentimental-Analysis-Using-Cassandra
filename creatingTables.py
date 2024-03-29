
import pandas as pd
from cassandra.cluster import Cluster
cluster=Cluster(["127.0.0.1"],port=9042)
session=cluster.connect("nosqlproject",wait_for_all_pools=True)
session.execute("DROP TABLE IF EXISTS nosqlproject.sap;")
session.execute("""CREATE TABLE nosqlproject.sap(sentiment text ,word text primary key );""")
session.execute(""COPY nosqlproject.sap(sentiment,word) from 'C:\\Users\\SRIHARISH\\Downloads\\preprocessedSA6.csv' with header =TRUE;"")