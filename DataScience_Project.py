import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from PIL import Image  # Added this import
import io

# def load_dataset():
#     """Load the pizza sales dataset."""
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         return pd.read_csv(uploaded_file)
#     return None

def load_dataset():
    """Load the pizza sales dataset from local file."""
    return pd.read_csv(r"D:\RVCE\1st sem\EL and projects 1st sem\DataScience\phase2\dataset\pizza_sales.csv")


def load_mask_image():
    """Load a mask image for word cloud."""
    uploaded_file = st.file_uploader("Choose a mask image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        return np.array(Image.open(uploaded_file))
    return None

def perform_eda(dataset):
    """Perform Exploratory Data Analysis."""
    st.subheader("Exploratory Data Analysis")
    
    # Outlier Treatment
    Q1 = dataset["total_price"].quantile(0.25)
    Q3 = dataset["total_price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = dataset[(dataset["total_price"] >= lower_bound) & (dataset["total_price"] <= upper_bound)]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total Price Distribution
    sns.histplot(df_cleaned['total_price'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Total Price Distribution")
    
    # Quantity Distribution
    sns.histplot(df_cleaned['quantity'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Quantity Distribution")
    
    # Correlation Heatmap
    numeric_df = df_cleaned.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title("Correlation Heatmap")
    
    # Pizza Category Count
    sns.countplot(x=df_cleaned["pizza_category"], ax=axes[1, 1])
    axes[1, 1].set_title("Pizza Category Frequency")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

def perform_clustering(dataset):
    """Perform K-Means Clustering."""
    st.subheader("K-Means Clustering")
    
    # Prepare data for clustering
    df_cleaned = dataset.dropna()
    
    # Encode pizza size
    size_mapping = {"S": 1, "M": 2, "L": 3, "XL": 4}
    df_cleaned["pizza_size"] = df_cleaned["pizza_size"].map(size_mapping)
    
    # Select features for clustering
    features = ["quantity", "total_price", "pizza_size"]
    df_cluster = df_cleaned[features].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    # Elbow Method to find optimal K
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plotting Elbow Method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow Method Plot
    ax1.plot(K_range, inertia, marker="o", linestyle="-")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method for Optimal K")
    
    # Optimal K (4 in this case)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(df_scaled)
    
    # Clustering Visualization
    scatter = ax2.scatter(
        df_cluster["quantity"],
        df_cluster["total_price"],
        c=df_cluster["Cluster"],
        cmap="viridis"
    )
    ax2.set_xlabel("Quantity")
    ax2.set_ylabel("Total Price")
    ax2.set_title("K-Means Clustering of Pizza Orders")
    ax2.legend(*scatter.legend_elements(), title="Clusters")
    
    plt.tight_layout()
    st.pyplot(fig)
    
 
def perform_decision_tree(dataset):
    """Perform Decision Tree Classification."""
    st.subheader("Decision Tree Classification")
    
    # Prepare data
    df_cleaned = dataset.dropna()
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df_cleaned["pizza_category_encoded"] = label_encoder.fit_transform(df_cleaned["pizza_category"])
    
    # Select features and target
    features = ["quantity", "total_price", "pizza_size"]
    X = df_cleaned[features]
    y = df_cleaned["pizza_category_encoded"]
    
    # Encode pizza size
    size_mapping = {"S": 1, "M": 2, "L": 3, "XL": 4}
    X["pizza_size"] = X["pizza_size"].map(size_mapping)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Create figure with larger size
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
    
    # Confusion Matrix Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Display Confusion Matrix
    st.pyplot(fig)
    
    # Decision Tree Plot (using the same figure)
    plt.figure(figsize=(25, 15))
    plot_tree(dt_model, 
              feature_names=features, 
              class_names=label_encoder.classes_, 
              filled=True, 
              fontsize=10,
              rounded=True,
              proportion=True)
    plt.title('Decision Tree Visualization', fontsize=16)
    plt.tight_layout()
    
    # Display Decision Tree
    st.pyplot(plt.gcf())
    
    # Show metrics 
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(class_report)

    
def generate_ingredient_wordcloud(dataset):
    """Generate Word Cloud for Ingredients."""
    st.subheader("Pizza Ingredients Word Cloud")
    
    # Combine all ingredients into a single string
    text = " ".join(dataset['pizza_ingredients'].dropna().str.lower().str.replace(",", ""))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        colormap="viridis"
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Common Pizza Ingredients")
    
    st.pyplot(fig)

def generate_pizza_name_wordcloud(dataset, mask_image=None):
    """Generate Word Cloud for Pizza Names."""
    st.subheader("Most Frequently Ordered Pizzas")
    
    # Remove "The" and "Pizza" from pizza names
    df_cleaned = dataset.copy()
    df_cleaned["pizza_name_cleaned"] = df_cleaned["pizza_name"].str.replace(r"\b(The|Pizza)\b", "", regex=True).str.strip()
    
    # Count frequency of cleaned pizza names
    pizza_counts = df_cleaned["pizza_name_cleaned"].value_counts().to_dict()
    
    # Check if mask image is loaded
    if mask_image is not None:
        wordcloud = WordCloud(
            width=800, height=800,
            background_color="white",
            colormap="autumn",
            mask=mask_image,
            contour_width=3,
            contour_color="brown"
        ).generate_from_frequencies(pizza_counts)
    else:
        # Fallback to standard wordcloud if no mask
        wordcloud = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="viridis"
        ).generate_from_frequencies(pizza_counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequently Ordered Pizzas", fontsize=14)
    
    st.pyplot(fig)
    
    
def show_dataset_overview(dataset):
    """Display basic dataset information."""
    st.subheader("📊 Dataset Overview")
    
    st.write("**Shape of the dataset:**", dataset.shape)
    st.write("**Column Names:**", list(dataset.columns))

    st.markdown("### First 5 Rows")
    st.dataframe(dataset.head())

    st.markdown("### Last 5 Rows")
    st.dataframe(dataset.tail())

    st.markdown("### Data Types and Non-Null Counts")
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.markdown("### Statistical Summary")
    st.dataframe(dataset.describe())

    st.markdown("### Missing Values")
    missing = dataset.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0])  # Display missing values only if any
    else:
        st.success("No missing values 🎉")  # Display a success message if no missing values
    #st.dataframe(missing[missing > 0] if missing.sum() > 0 else "No missing values 🎉")

    st.markdown("### Data Types")
    st.dataframe(dataset.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

    
    
    

def main():
    st.set_page_config(page_title="🍕 Pizza Sales Analysis Tool", layout="wide")
    st.title("🍕 Pizza Sales Analysis Tool")
    
    # Load dataset
    dataset = load_dataset()
    mask_image = None
    
    


    if dataset is not None:
        # Sidebar for analysis selection
        analysis_option = st.sidebar.selectbox(
            "Select Analysis",
            [
                "Dataset Overview",
                "Exploratory Data Analysis", 
                "K-Means Clustering", 
                "Decision Tree Classification", 
                "Ingredient Word Cloud", 
                "Pizza Name Word Cloud"
            ]
        )
        
        if analysis_option == "Dataset Overview":
            show_dataset_overview(dataset)

        # Optional mask image upload for Pizza Name Word Cloud
        if analysis_option == "Pizza Name Word Cloud":
            mask_image = load_mask_image()

        # Perform selected analysis
        if analysis_option == "Exploratory Data Analysis":
            perform_eda(dataset)
        elif analysis_option == "K-Means Clustering":
            perform_clustering(dataset)
        elif analysis_option == "Decision Tree Classification":
            perform_decision_tree(dataset)
        elif analysis_option == "Ingredient Word Cloud":
            generate_ingredient_wordcloud(dataset)
        elif analysis_option == "Pizza Name Word Cloud":
            generate_pizza_name_wordcloud(dataset, mask_image)

if __name__ == "__main__":
    main()