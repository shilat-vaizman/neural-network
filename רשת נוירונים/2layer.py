import numpy as np
import os

# פונקציית sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# פונקציית המגזרת של sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# קביעת מספר האיטרציות לאימון
num_iterations = 20000

# הגדרת פונקציה לאימון ובדיקה של הרשת עם שתי שכבות חבויות
def train_and_evaluate_two_hidden_layers(num_train_groups, x_data, y_data, x_test_new, y_test_new):
    input_neurons = x_data.shape[1]
    hidden_neurons_1 = 20  # גודל הנוירונים בשכבה הראשונה
    hidden_neurons_2 = 10  # גודל הנוירונים בשכבה השנייה
    output_neurons = 1

    # Initialize weights with small random values
    np.random.seed(1)
    synapse_input_to_hidden1 = 2 * np.random.random((input_neurons, hidden_neurons_1)) - 1
    synapse_hidden1_to_hidden2 = 2 * np.random.random((hidden_neurons_1, hidden_neurons_2)) - 1
    synapse_hidden2_to_output = 2 * np.random.random((hidden_neurons_2, output_neurons)) - 1

    results = []
    for i in range(num_iterations):
        # Forward propagation
        layer_1 = sigmoid(np.dot(x_data[:num_train_groups], synapse_input_to_hidden1))
        layer_2 = sigmoid(np.dot(layer_1, synapse_hidden1_to_hidden2))
        output = sigmoid(np.dot(layer_2, synapse_hidden2_to_output))

        # Calculate error
        output_error = y_data[:num_train_groups] - output
        mean_error = np.mean(np.abs(output_error))

        # Save results
        iteration_result = {
            'iteration': i + 1,
            'train_success_rate': (1 - mean_error) * 100
        }
        results.append(iteration_result)

        # Stop condition
        if (1 - mean_error) * 100 >= 90:
            break

        # Backpropagation
        output_delta = output_error * sigmoid_derivative(output)
        layer_2_error = output_delta.dot(synapse_hidden2_to_output.T)
        layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_hidden1_to_hidden2.T)
        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

        # Update weights
        synapse_hidden2_to_output += layer_2.T.dot(output_delta)
        synapse_hidden1_to_hidden2 += layer_1.T.dot(layer_2_delta)
        synapse_input_to_hidden1 += x_data[:num_train_groups].T.dot(layer_1_delta)

    # Testing with new test set
    layer_1_test_new = sigmoid(np.dot(x_test_new, synapse_input_to_hidden1))
    layer_2_test_new = sigmoid(np.dot(layer_1_test_new, synapse_hidden1_to_hidden2))
    output_test_new = sigmoid(np.dot(layer_2_test_new, synapse_hidden2_to_output))
    output_error_test_new = y_test_new - output_test_new
    mean_error_test_new = np.mean(np.abs(output_error_test_new))
    test_success_rate = (1 - mean_error_test_new) * 100

    return synapse_input_to_hidden1, synapse_hidden1_to_hidden2, synapse_hidden2_to_output, results, test_success_rate

# יצירת נתונים אקראיים
x_data = np.random.rand(20, 100)  # 20 קבוצות נתונים, כל אחת עם 100 תכונות
y_data = np.random.randint(0, 2, size=(20, 1))  # 20 קבוצות תוצאות

# קבוצות בדיקה חדשות לבדיקה
x_test_new = np.random.rand(10, 100)  # לדוגמה, 10 קבוצות בדיקה חדשות עם 100 תכונות
y_test_new = np.random.randint(0, 2, size=(10, 1))  # תוצאות לקבוצות הבדיקה החדשות

# רשימת קבוצות הלמידה לבדיקה
train_group_counts = [5, 10, 20]

# שם קובץ הטקסט
filename = 'summary_Two_hidden_layers_with_multiple_tests.txt'

# פתיחת קובץ הטקסט לכתיבה
with open(filename, 'w') as f:
    f.write("Training and Testing Summary\n")
    f.write("===========================\n\n")

    for num_train_groups in train_group_counts:
        f.write(f"Training with {num_train_groups} groups\n")
        f.write("-------------------------------\n")

        synapse_input_to_hidden1, synapse_hidden1_to_hidden2, synapse_hidden2_to_output, results, test_success_rate = train_and_evaluate_two_hidden_layers(num_train_groups, x_data, y_data, x_test_new, y_test_new)

        for result in results:
            f.write(f"Iteration: {result['iteration']}\n")
            f.write(f"Training Success Rate: {result['train_success_rate']}%\n")
            f.write("\n")

            # Check if success rate is 90% or higher
            if result['train_success_rate'] >= 90:
                f.write(f"Reached 90% success rate at iteration {result['iteration']}\n")
                f.write("\n")

        f.write(f"Testing with new test set\n")
        f.write("-------------------------------\n")
        f.write(f"Test Success Rate for new test set: {test_success_rate}%\n")
        f.write("\n")

print(f"Text file created successfully at {os.path.abspath(filename)}")
