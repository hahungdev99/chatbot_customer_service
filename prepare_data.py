import pandas as pd

def clean_csv(input_file, output_file, remove_value="Không tìm thấy trang"):
    """
    Load a CSV file, remove rows with a specific value in the 'name' column, and save the cleaned data.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the cleaned CSV file.
    :param remove_value: The value to remove from the 'name' column.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Check if the 'name' column exists
        if 'name' not in df.columns:
            raise ValueError("The 'name' column is missing in the CSV file.")

        # Filter out rows where the 'name' column matches the remove_value
        cleaned_df = df[df['name'] != remove_value]

        # Save the cleaned DataFrame to a new CSV file
        cleaned_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Cleaned data saved to: {output_file}")

    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Specify the input and output file paths
    input_csv_path = "datasets/products_origin.csv"  # Replace with your input file path
    output_csv_path = "datasets/products_origin.csv"  # Replace with your desired output file path

    # Clean the CSV file
    clean_csv(input_csv_path, output_csv_path)


import pandas as pd

def remove_column(input_file, output_file, column_to_remove):
    """
    Load a CSV file, remove a specified column, and save the updated data.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the updated CSV file.
    :param column_to_remove: Name of the column to remove.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Check if the column exists in the DataFrame
        if column_to_remove not in df.columns:
            raise ValueError(f"The column '{column_to_remove}' does not exist in the CSV file.")

        # Drop the specified column
        df.drop(columns=[column_to_remove], inplace=True)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Updated data saved to: {output_file}")

    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Specify the input and output file paths
    input_csv_path = "datasets/products_origin.csv" 
    output_csv_path = "datasets/products_origin.csv" 

    # Specify the column to remove
    column_to_remove = "information_product_embedding"

    # Remove the column and save the updated file
    remove_column(input_csv_path, output_csv_path, column_to_remove)