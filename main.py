import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Import ML libraries ---
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==============================================================================
# PART 1: IMAGE PROCESSING (Modified for VS Code)
# ==============================================================================
# CONFIGURATION: Change this string to match your image filename exactly
image_filename = "heart.jpg" 

print(f"Loading image: {image_filename}...")
img = cv2.imread(image_filename)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not find '{image_filename}'. Please ensure the file is in the same folder as this script.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
v = np.median(gray)
sigma = 0.33
lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(image=gray, threshold1=lower_thresh, threshold2=upper_thresh)
kernel = np.ones((5,5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, hierarchy = cv2.findContours(
    closed_edges,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

# --- Logic to keep top 20 contours (or all if less than 20) ---
max_contours = 20
min_points = 20
significant_contours = [cnt for cnt in contours if len(cnt) >= min_points]
sorted_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)

if len(sorted_contours) > max_contours:
    filtered_contours = sorted_contours[:max_contours]
    print(f"Found {len(sorted_contours)} significant contours, keeping the largest {max_contours}.")
else:
    filtered_contours = sorted_contours
    print(f"Found {len(sorted_contours)} significant contours, keeping all of them.")

# ==============================================================================
# PART 2: ML-ORIENTED MODEL FITTING (Using Scikit-learn)
# ==============================================================================
# (This part is identical to your Colab code, copy it exactly as it was)

all_models_to_plot = []

for contour in filtered_contours:
    points = contour.reshape(-1, 2)
    segment_length = 15
    polynomial_degree = 3

    if len(points) > polynomial_degree:
        segments = []
        for i in range(0, len(points), segment_length - 1):
            segment = points[i:i + segment_length]
            if len(segment) > polynomial_degree:
                segments.append(segment)

        for segment in segments:
            x0, y0 = segment[0]
            t_train = np.linspace(0, 1, len(segment)).reshape(-1, 1)
            y_train_x = segment[:, 0] - x0
            y_train_y = segment[:, 1] - y0

            model_x = make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression())
            model_x.fit(t_train, y_train_x)

            model_y = make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression())
            model_y.fit(t_train, y_train_y)

            all_models_to_plot.append((model_x, x0, model_y, y0))

# ==============================================================================
# PART 3: PLOTTING
# ==============================================================================
print(f"Plotting {len(all_models_to_plot)} total segments...")
plt.figure(figsize=(10, 10))
ax = plt.gca()

for model_x, x0, model_y, y0 in all_models_to_plot:
    t_vals_predict = np.linspace(0, 1, 100).reshape(-1, 1)
    
    x_coords_shape = model_x.predict(t_vals_predict)
    y_coords_shape = model_y.predict(t_vals_predict)
    
    x_coords_final = x_coords_shape + x0
    y_coords_final = y_coords_shape + y0
    
    ax.plot(x_coords_final, y_coords_final, 'b-')

ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
plt.title(f"Plot of ALL Segments from Top Contours (ML Method)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()