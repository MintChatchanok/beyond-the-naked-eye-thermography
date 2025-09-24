!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

def load_infrared_thermography_temperature_data():
    # Fetch dataset from UCI ML Repo
    dataset = fetch_ucirepo(id=925)

    # Extract features and targets
    X = dataset.data.features
    y = dataset.data.targets

    # Combine into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    return df
df = load_infrared_thermography_temperature_data()
print(df.head())  # Display first five rows
print(df.info())  # Summary of dataset
print(df.describe())  # Basic statistics

df.to_csv("infrared_thermography_temperature.csv", index=False)
