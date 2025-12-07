{"id": "58320", "variant": "standard"}
# Cat vs Dog Image Classification (CNN + Gradio Demo)

This project trains a Convolutional Neural Network (CNN) to classify images as either **cat** or **dog**, using a custom dataset stored locally. The final model is deployed through a simple **Gradio** web interface for easy testing.

---

##  Project Structure

```
catOrdog/
│
├── src/
│   ├── train_cnn.py
│   ├── predict_single.py
│   └── demo_gradio.py
│
├── saved_model/        # generated after training (ignored in Git)
├── results/            # accuracy/loss plots (ignored in Git)
├── dataset/            # not included; added locally by user
│
├── requirements.txt
├── README.md
└── report.md
```

---

##  Installing Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Use a virtual environment if you want to keep things clean.

---

##  Adding the Dataset (IMPORTANT)

The dataset is **not** in the repo.

1. Download `dataset.zip` from the shared link (Google Drive, etc.)
2. Extract it into the project root so it looks like this:

```
dataset/
    training_set/
        cats/
        dogs/
    test_set/
        cats/
        dogs/
```

---

##  Training the Model

Run:

```bash
python src/train_cnn.py
```

This will:

- Train the CNN  
- Generate accuracy/loss plots  
- Save the trained model into `saved_model/`

---

##  Predicting a Single Image

```bash
python src/predict_single.py --img path/to/image.jpg
```

Outputs the predicted class.

---

##  Running the Gradio Demo

To open a simple web UI in your browser:

```bash
python src/demo_gradio.py
```

Upload an image → see prediction instantly.

---

##  Requirements

Python 3.8+  
TensorFlow / Keras  
Gradio  
NumPy  
Pillow  
Matplotlib

(All included in `requirements.txt`)
