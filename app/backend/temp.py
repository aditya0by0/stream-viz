def get_metric_score_list():

    file_path = "list.txt"

    # Initialize an empty list to store the retrieved data
    retrieved_data = []

    # Read the list from the text file
    with open(file_path, "r") as file:
        for line in file:
            retrieved_data.append(line.strip())

    # print(retrieved_data)
    return retrieved_data
