import numpy as np
import matplotlib.pyplot as plt
import random


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
    """
    פונקציה שמאריכה או מקצרת את הקלט לאורך קבוע.
    """
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


# יצירת נתונים אקראיים
x_data, y_data = create_data(20)  # 20 קבוצות נתונים

# הדפסת ממדים לאימות
print(x_data.shape)  # צריך להיות (60, 200)
print(y_data.shape)  # צריך להיות (60, 1)
