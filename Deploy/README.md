# Getting started in 10 minutes
Clone this repo

Install requirements

Run the script

Check http://localhost:5000

Done! 🎉
________________________________________
# Installation

## Clone the repo

$ git clone https://github.com/mtobeiyf/keras-flask-deploy-webapp.git

## Install requirements

$ pip install -r requirements.txt

Make sure you have the following installed:

- tensorflow

- keras

- flask

- pillow

- h5py

- gevent

## Run with Python

Python 2.7 or 3.5+ are supported and tested.

$ python app.py

## Play

Open http://localhost:5000 and have fun. 

## Customization

### Use your own model

Place your trained .h5 file saved by model.save() under models directory.

### Use other pre-trained model

See Keras applications for more available models such as DenseNet, MobilNet, NASNet, etc.

### UI Modification

Modify files in templates and static directory.
index.html for the UI and main.js for all the behaviors

## Deployment

To deploy it for public use, you need to have a public linux server.

### Run the app

Run the script and hide it in background with tmux or screen.

$ python app.py

You can also use gunicorn instead of gevent

$ gunicorn -b 127.0.0.1:5000 app:app

### Set up Nginx

To redirect the traffic to your local app. Configure your Nginx .conf file.

server {
    listen  80;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
    }
}

## More resources

Check Siraj's "How to Deploy a Keras Model to Production" video. The corresponding repo.

Building a simple Keras + deep learning REST API
