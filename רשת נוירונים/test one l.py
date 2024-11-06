import numpy as np
import os
import random


# פונקציית sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# פונקציית המגזרת של sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# פונקציית relu
def relu(x):
    return np.maximum(0, x)


# פונקציית המגזרת של relu
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# פונקציית Tanh
def tanh(x):
    return np.tanh(x)


# פונקציית המגזרת של Tanh
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# קביעת מספר האיטרציות לאימון
num_iterations = 20000


# הגדרת פונקציה לאימון ובדיקה של הרשת
def train_and_evaluate(num_train_groups, x_data, y_data):
    # נתוני האימון
    x_train = x_data[:num_train_groups]
    y_train = y_data[:num_train_groups]

    # נתוני הבדיקה (האחרונה היא קבוצת הבדיקה)
    x_test = x_data[num_train_groups:num_train_groups + 1]
    y_test = y_data[num_train_groups:num_train_groups + 1]

    # הגדרת משקלי הסינפסים בין השכבות
    input_neurons = x_train.shape[1]  # מספר הנוירונים בשכבת הקלט
    hidden_neurons = 20  # מספר הנוירונים בשכבה המוסתרת הראשונה
    hidden_neurons_2 = 20  # מספר הנוירונים בשכבה המוסתרת השנייה
    output_neurons = 1  # מספר הנוירונים בשכבת הפלט

    # יצירת מטריצות הסינפסים עם ערכים אקראיים קטנים
    np.random.seed(1)
    synapse_one_to_two = 2 * np.random.random((input_neurons, hidden_neurons)) - 1
    synapse_two_to_three = 2 * np.random.random((hidden_neurons, hidden_neurons_2)) - 1
    synapse_three_to_output = 2 * np.random.random((hidden_neurons_2, output_neurons)) - 1

    # רשימת נתונים לשמירה עבור קובץ הטקסט
    results = []

    for i in range(num_iterations):
        # פורווארד פרופגיישן
        layer_1 = tanh(np.dot(x_train, synapse_one_to_two))
        layer_2 = tanh(np.dot(layer_1, synapse_two_to_three))
        output = sigmoid(np.dot(layer_2, synapse_three_to_output))

        # חישוב שגיאת הפלט
        output_error = y_train - output

        # חישוב השגיאה הממוצעת
        mean_error = np.mean(np.abs(output_error))

        # שמירת התוצאות
        iteration_result = {
            'iteration': i + 1,
            'train_success_rate': (1 - mean_error) * 100
        }
        results.append(iteration_result)

        # תנאי עצירה: אם הצליח בכדי לבדוק את נתוני הבדיקה
        if (1 - mean_error) * 100 >= 90:
            layer_1_test = tanh(np.dot(x_test, synapse_one_to_two))
            layer_2_test = tanh(np.dot(layer_1_test, synapse_two_to_three))
            output_test = sigmoid(np.dot(layer_2_test, synapse_three_to_output))

            # חישוב שגיאת הפלט לצורך בדיקת הביצועים
            output_error_test = y_test - output_test
            mean_error_test = np.mean(np.abs(output_error_test))

            # שמירת תוצאות הבדיקה
            iteration_result.update({
                'test_success_rate': (1 - mean_error_test) * 100,
                'finished_at': i + 1
            })
            break

        # בקידום אחורי: חישוב שגיאות ועדכון המשקלים
        output_delta = output_error * sigmoid_derivative(output)
        layer_2_error = output_delta.dot(synapse_three_to_output.T)
        layer_2_delta = layer_2_error * tanh_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_two_to_three.T)
        layer_1_delta = layer_1_error * tanh_derivative(layer_1)

        synapse_three_to_output += np.dot(layer_2.T, output_delta)
        synapse_two_to_three += np.dot(layer_1.T, layer_2_delta)
        synapse_one_to_two += np.dot(x_train.T, layer_1_delta)

    return results


# יצירת הנתונים החדשים
def create_circle(radius, center, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def create_triangle(center, size):
    height = size * (np.sqrt(3) / 2)
    vertices = [(center[0] - size / 2, center[1] - height / 3),
                (center[0] + size / 2, center[1] - height / 3),
                (center[0], center[1] + 2 * height / 3)]
    x = [v[0] for v in vertices] + [vertices[0][0]]
    y = [v[1] for v in vertices] + [vertices[0][1]]
    return x, y


def create_ellipse(a, b, center, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + a * np.cos(theta)
    y = center[1] + b * np.sin(theta)
    return x, y


def create_shapes_group():
    circle_radius = random.uniform(0.5, 1.5)
    ellipse_a = random.uniform(1, 2)
    ellipse_b = random.uniform(0.5, 1.5)
    triangle_size = random.uniform(1, 2)
    center_x = random.uniform(1, 9)
    center_y = random.uniform(1, 9)
    center = (center_x, center_y)

    circle_x, circle_y = create_circle(circle_radius, center)
    ellipse_x, ellipse_y = create_ellipse(ellipse_a, ellipse_b, center)
    triangle_x, triangle_y = create_triangle(center, triangle_size)
    return (circle_x, circle_y), (ellipse_x, ellipse_y), (triangle_x, triangle_y)


def pad_or_truncate(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        return np.pad(data, (0, target_length - len(data)), 'constant')
    return data


def create_data(num_groups, num_points_per_shape=100):
    data = []
    labels = []
    target_length = num_points_per_shape * 2  # 100 x and 100 y coordinates

    for _ in range(num_groups):
        circle, ellipse, triangle = create_shapes_group()
        circle_data = np.concatenate((circle[0], circle[1]))
        ellipse_data = np.concatenate((ellipse[0], ellipse[1]))
        triangle_data = np.concatenate((triangle[0], triangle[1]))

        data.append(pad_or_truncate(circle_data, target_length))  # Circle
        labels.append(0)

        data.append(pad_or_truncate(ellipse_data, target_length))  # Ellipse
        labels.append(1)

        data.append(pad_or_truncate(triangle_data, target_length))  # Triangle
        labels.append(2)

    return np.array(data), np.array(labels).reshape(-1, 1)


# נורמליזציה של הנתונים
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


# יצירת נתונים אקראיים
x_data, y_data = create_data(20)  # 20 קבוצות נתונים

# נורמליזציה של הנתונים
x_data = normalize_data(x_data)

# רשימת מספרי קבוצות האימון לבדיקות שונות
train_group_counts = [5, 10, 19]

# יצירת קובץ טקסט והוספת תוכן
filename = 'summary_Two_hidden_layers_tanh.txt'
with open(filename, 'w') as f:
    f.write("Training Summary\n")
    f.write("================\n\n")

    for num_train_groups in train_group_counts:
        f.write(f"Training with {num_train_groups} groups\n")
        f.write("-------------------------------\n")

        results = train_and_evaluate(num_train_groups, x_data, y_data)
        for result in results:
            f.write(f"Iteration: {result['iteration']}\n")
            f.write(f"Training Success Rate: {result['train_success_rate']}%\n")
            if 'test_success_rate' in result:
                f.write(f"Test Success Rate: {result['test_success_rate']}%\n")
                f.write(f"Finished at Iteration: {result['finished_at']}\n")
            f.write("\n")
        f.write("\n")
