import os
import subprocess
import csv
import time
import re

def extract_accuracy_from_output(output):
    """
    Extract the test accuracy from the output of eval.py script.

    Args:
    output (str): The output string from eval.py.

    Returns:
    float: Extracted test accuracy, or 'N/A' if not found.
    """
    # Regular expression pattern to match the test accuracy
    accuracy_pattern = r"Test accuracy: ([\d.]+)%"

    # Search for the pattern in the output
    match = re.search(accuracy_pattern, output.decode('utf-8'))

    # Extract and return the accuracy if found
    if match:
        return float(match.group(1))
    else:
        return 'N/A'

iterations = range(28)  # 0 to 27
blocks = range(7)  # 0 to 6
data = []
base_path = "/home/ek3296/netadapt"
accuracy_path = os.path.join(base_path, "models/alexnet/prune-by-mac/worker")
resource_path = os.path.join(base_path, "models/alexnet/prune-by-mac/worker")
model_path = os.path.join(base_path, "models/alexnet/prune-by-mac/master")

for iter in iterations:
    for block in blocks:
        # Read accuracy and resource files
        if (iter<1):
            break
    
        accuracy_file_path = os.path.join(accuracy_path, f'iter_{iter}_block_{block}_accuracy.txt')
        resource_file_path = os.path.join(resource_path, f'iter_{iter}_block_{block}_resource.txt')

        with open(accuracy_file_path, 'r') as file:
            accuracy = file.read().strip()

        with open(resource_file_path, 'r') as file:
            resource = file.read().strip()

        # Get model file size
        model_file = f'models/alexnet/prune-by-mac/master/iter_{iter}_best_model.pth.tar'
        model_size = os.path.getsize(model_file)

        # Evaluate test data
        start_time = time.time()
        result = subprocess.run(['python', 'eval.py', 'data/', '--dir', model_file, '--arch', 'alexnet'], capture_output=True)
        end_time = time.time()
        test_accuracy = extract_accuracy_from_output(result.stdout)  # You need to implement this function
        eval_time = end_time - start_time

        data_row = [iter, block, accuracy, resource, model_size, test_accuracy, eval_time]
        data.append(data_row)

        # Print the data for each evaluation
        print(f"Iteration: {iter}, Block: {block}, Training Accuracy: {accuracy}, Resource Used: {resource}, Model Size: {model_size}, Test Accuracy: {test_accuracy}, Eval Time: {eval_time}")

# Write to CSV
with open('model_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'Block', 'Training Accuracy', 'Resource Used', 'Model Size', 'Test Accuracy', 'Eval Time'])
    writer.writerows(data)

