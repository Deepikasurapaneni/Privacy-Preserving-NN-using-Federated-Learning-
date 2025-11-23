from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

from MLModel import ScatterLinear, get_scatter_transform

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a strong key in production

MODEL_DIR = 'models'

CLASS_NAMES = {
    'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'fashionmnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
}

def load_model(dataset, clients):
    path = f"{MODEL_DIR}/{dataset}_{clients}clients.pth"
    scattering, K, (h, w) = get_scatter_transform()
    model = ScatterLinear(in_channels=K, hw_dims=(h, w), classes=10)

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("global_model."):
            new_state_dict[k.replace("global_model.", "")] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model, scattering

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))
USERS = {
    'deepika': 'password123',
    'admin': 'adminpass'
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    dataset = request.form.get('dataset')
    clients = request.form.get('clients')
    image_file = request.files.get('image')

    if not all([dataset, clients, image_file]):
        return jsonify({'error': 'Missing input'}), 400

    try:
        model, scattering = load_model(dataset, clients)
        image = Image.open(image_file).convert('L')

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        tensor_image = transform(image).unsqueeze(0)

        with torch.no_grad():
            scatter_features = scattering(tensor_image)
            output = model(scatter_features)
            pred_index = torch.argmax(output, 1).item()

        class_list = CLASS_NAMES.get(dataset, [])
        class_name = class_list[pred_index] if pred_index < len(class_list) else "Unknown"

        return render_template('result.html', prediction=class_name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
