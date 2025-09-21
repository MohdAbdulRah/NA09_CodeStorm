training_model.h5 link:- https://drive.google.com/file/d/1OfAiw-D6W5aTEVcaTqytJuf7oz-vgYvx/view?usp=sharing

traning_model.h5 will be get after running the Train_Plant_Disease.ipynb

# 🌱 Model1_Accuracy95% — Plant Disease Detection

This folder contains **Model1_Accuracy95%**, a machine learning model capable of detecting plant diseases from leaf images, with ~**95% accuracy**. Part of the **NA09_CodeStorm** project by MohdAbdulRah.

---

## 📋 Contents

| File / Folder              | Description                                   |
|----------------------------|-----------------------------------------------|
| `training_model.h5`        | The trained model file for disease detection. |
| `training_hist.json`       | It stores the loss                            |
| `README.md`                | This documentation.                           |
| `Train_Plant_Disease.ipynb`| To Train a model                              |
| `Test_Plant_Disease.ipynb` | To Test a model                               |
|  'Train Folder'            | Train Images                                  |
|  'Valid'                   | Validation Images                             |
| 'Test'                     | Test Images                                   |
------------------------------------------------------------------------------

## 🚀 Usage

Here are the steps to use this model (assuming integration into a larger app or script):

1. **Preprocess an image**  
   - Resize / rescale leaf image to match model’s input shape (e.g., 224×224 or 256×256).  
   - Normalize pixel values (e.g. /255), possibly apply augmentations if needed.

2. **Load the model**  
   ```python
   from tensorflow.keras.models import load_model

   model = load_model("Model1_Accuracy95%/model1.h5")
Predict

python
Copy code
import numpy as np
from utils import preprocess_image, load_class_names

img = preprocess_image("path/to/leaf.jpg")
preds = model.predict(np.expand_dims(img, axis=0))
class_names = load_class_names("Model1_Accuracy95%/training_hist.json")

predicted_class = class_names[np.argmax(preds)]
confidence = float(np.max(preds))
print(f"Disease: {predicted_class}, Confidence: {confidence:.2f}")

### 🔍 Model Details
Architecture: Convolutional Neural Network (CNN)


Accuracy: ~95% on the test set

Loss Function: (e.g., categorical crossentropy)

Optimizer: (e.g., Adam)

Input Size: (e.g., 224 × 224 RGB)

### 📈 Performance & Metrics
Include relevant metrics such as:

Confusion matrix

Precision / Recall / F1-Score for each class

Overall accuracy

Maybe ROC-AUC if applicable


### ⚙️ Dependencies
Here are some likely Python libraries needed:

text
Copy code
tensorflow or keras
numpy
Pillow or OpenCV
scikit-learn (for metrics)
matplotlib or seaborn (for visualizations)
You can install required dependencies via:

bash
Copy code
pip install -r requirements.txt

## Dataset Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### 💡 Future Work
Ideas to improve or expand the model:

Add more disease classes / plant species

Improve performance for low-quality images (blurry, shadows)

Build a mobile-friendly lightweight version

Integrate with a web UI or Streamlit app for live use

Provide feedback & suggestions (treatments / prevention)

###  Contribution
Contributions are welcome! If you'd like to improve the model, add features, or fix issues:

Fork the repository

Create a new branch with your feature or fix

Submit a Pull Request with clear description of what you changed



### 👤 Author
Mohd Abdul Rahman — Creator & Maintainer
GitHub: @MohdAbdulRah
