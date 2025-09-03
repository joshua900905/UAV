import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # --- Step 1: Read and prepare the data ---
    print("Reading 'analysis_log.csv'...")
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("analysis_log.csv")
    
    # Sort the data by timestep to ensure the line plot connects points in chronological order
    df = df.sort_values(by='timestep').reset_index(drop=True)

    # --- Step 2: Filter the data to keep only the 'search' strategy ---
    print("Filtering data to keep 'search' strategy only...")
    df_search_only = df[df['strategy'] == 'search'].copy()
    
    # Print a summary of the filtering result
    print(f"Original data has {len(df)} rows. Filtered data has {len(df_search_only)} rows.")

    # Set the visual style of the plots
    sns.set_style("whitegrid")

    # --- Step 3: Plot 'Tree Length' for the 'search' strategy ---
    print("\nGenerating Plot 1: Tree Length Over Time (Search Strategy Only)...")
    plt.figure(figsize=(15, 7))

    # Plot the data using the filtered DataFrame
    sns.lineplot(data=df_search_only, x="timestep", y="tree_length", color='darkorange')

    # Add English titles and labels
    plt.title("Tree Length Over Time (Search Strategy Only)", fontsize=16)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Tree Length", fontsize=12)
    
    # Display the plot
    plt.show()

    # --- Step 4: Plot 'Relay Count' for the 'search' strategy ---
    print("\nGenerating Plot 2: Relay Count Over Time (Search Strategy Only)...")
    plt.figure(figsize=(15, 7))
    
    # Plot the data using the same filtered DataFrame
    sns.lineplot(data=df_search_only, x="timestep", y="relay_count", color='darkorange')
    
    # Add English titles and labels
    plt.title("Relay Count Over Time (Search Strategy Only)", fontsize=16)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Relay Count", fontsize=12)

    # Display the plot
    plt.show()

    print("\nAll plots generated successfully!")

# --- Error Handling ---
except FileNotFoundError:
    print("\nError: 'analysis_log.csv' not found.")
    print("Please ensure the Python script and the CSV file are in the same directory.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")