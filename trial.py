import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_resource
def load_data():
    return pd.read_csv("Country_data.csv")

dataset = load_data()

# Preprocess data
def preprocess_data(df):
    df['exports'] = (df['exports']/100)*df['gdpp']
    df['health'] = (df['health']/100)*df['gdpp']
    df['imports'] = (df['imports']/100)*df['gdpp']
    return df

dataset = preprocess_data(dataset)

# Perform clustering
def perform_clustering(df, num_clusters, method):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    if len(df) == 1:
        # If only one sample provided, return default cluster label
        return pd.DataFrame({'cluster_id': [0]})
    
    if method == 'KMeans':
        km = KMeans(n_clusters=num_clusters, max_iter=100, random_state=101)
        km.fit(scaled_data)
        cluster_labels = km.labels_
        return pd.DataFrame({'gdpp': df[:, 0], 'child_mort': df[:, 1], 'cluster_id': cluster_labels})
    elif method == 'Hierarchical':
        mergings = linkage(scaled_data, method="complete", metric='euclidean')
        cluster_labels = cut_tree(mergings, n_clusters=num_clusters).reshape(-1, )
        return pd.DataFrame({'gdpp': df[:, 0], 'child_mort': df[:, 1], 'cluster_id': cluster_labels})

# Streamlit App
st.title('Country Clustering Visualization')

# Input Section
st.sidebar.subheader('Input Parameters')
selected_country = st.sidebar.selectbox('Select a country', dataset['country'])
gdpp = st.sidebar.number_input('GDP per capita (gdpp)', value=0)
income = st.sidebar.number_input('Net Income per person (Income)', value=0)
child_mort = st.sidebar.number_input('Child Mortality (child_mort)', value=0)

if st.sidebar.button('Submit'):
    # Display whether the country is developed, underdeveloped, or developing
    country_data = pd.DataFrame([[gdpp, income, child_mort]], columns=['gdpp', 'income', 'child_mort'])
    country_cluster = perform_clustering(country_data.values, 3, 'KMeans')
    cluster_label = country_cluster.iloc[0]['cluster_id']
    if cluster_label == 0:
        st.write(f"{selected_country} is an underdeveloped country.")
    elif cluster_label == 1:
        st.write(f"{selected_country} is a developing country.")
    elif cluster_label == 2:
        st.write(f"{selected_country} is a developed country.")

# Clustering Method Selection
st.sidebar.subheader('Clustering Method Selection')
clustering_method = st.sidebar.radio('Select Clustering Method', ['KMeans', 'Hierarchical'])
num_clusters = st.sidebar.number_input('Number of Clusters', value=3)

# Display Graphs
if st.sidebar.button('Generate Clustering Graphs'):
    st.subheader('Clustering Visualization')
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    if clustering_method == 'KMeans':
        clustered_data = perform_clustering(dataset[['gdpp', 'child_mort']].values, num_clusters, 'KMeans')
        sns.scatterplot(x='gdpp', y='child_mort', hue='cluster_id', palette='Set1', data=clustered_data, ax=ax[0, 0])
        sns.boxplot(x='cluster_id', y='gdpp', data=clustered_data, ax=ax[0, 1])
        sns.boxplot(x='cluster_id', y='child_mort', data=clustered_data, ax=ax[1, 0])
        sns.countplot(x='cluster_id', data=clustered_data, ax=ax[1, 1])
        plt.tight_layout()
        st.pyplot(fig)
    elif clustering_method == 'Hierarchical':
        clustered_data = perform_clustering(dataset[['gdpp', 'child_mort']].values, num_clusters, 'Hierarchical')
        sns.scatterplot(x='gdpp', y='child_mort', hue='cluster_id', palette='Set1', data=clustered_data, ax=ax[0, 0])
        sns.boxplot(x='cluster_id', y='gdpp', data=clustered_data, ax=ax[0, 1])
        sns.boxplot(x='cluster_id', y='child_mort', data=clustered_data, ax=ax[1, 0])
        sns.countplot(x='cluster_id', data=clustered_data, ax=ax[1, 1])
        plt.tight_layout()
        st.pyplot(fig)
