'''
Created on 28 May 2016

@author: Matti Kankkonen
'''
import os
from flask import Flask
from flask import request

import urllib
import numpy as np
import re
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

app = Flask(__name__)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
sess = tf.Session()

def train_tensor():
    ''' This is the Google's MNIST example from the TensorFlow
    web site. Kind a of Hello World for AI '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

def evaluate_image(dir):
    orig = Image.open(dir + "/img.jpg")
    sized = orig.resize([28,28])
    im = sized.convert("L")
    col, row = im.size
    data = np.zeros((1,row*col))
     
    pixels = im.load()
    
    for i in range(row):
        for j in range(col):
            value = pixels[j,i]
            data[0,i*col + j] = value
                
    
    ''' This is too rough currently pic can't be just 1's and 0's 
    PILLOW probably would have some fancy existing function for this '''
    for i in range(row):
        row_str = []
        for j in range(col):
            value = (255 - data[0,i*col + j])
            if( value > 200 ):
                value = 1
            elif (value > 110 ):
                value = 1
            else: 
                value = 0
            row_str.append(value)
            data[0,i*col + j] = value
        
        
    im.save(dir + "/latest_sample.jpg")
    ''' The training is done and now we should try to use 
    the model to recognise our picture. '''
    prediction = tf.argmax(y,1)
    probabilities = y
    #print "Probabilities", probabilities.eval(feed_dict={x: data}, session=sess)
    return prediction.eval(feed_dict={x: data}, session=sess)[0].astype(str)
 

@app.route("/env")
def list_env():
    response_body = ['%s: %s' % (key, value)for key, value in sorted(app.config.items())]
    response_body = '\n'.join(response_body)
    return response_body
        
@app.route("/")
def hello():
    
    imgurl=request.args.get('imgurl')
    usejson=request.args.get('json')    
    if (imgurl != None):
        dir = "/tmp"
        
        # TODO: urllib is handy but doesn't tell anything 
        # about success so currently if we get invalid URL 
        # we simply crash. 
        #
        # URL handling also lacking user input sanitation!!
        # Of course not to mention the concurrency support
        # or more like the lack of. It's just currently
        # very convenient to save the images to file for debugging
        urllib.urlretrieve(imgurl, dir+"/img.jpg")
        result = "invalid"
        result = evaluate_image(dir)
        
        if (usejson != None):
            message = '{"evaluation":{ "status":'
            if (result != "invalid"):
                message = message + '"valid",'
            else:
                message = message + '"invalid",'
            
            message = message + '"url":"'+ imgurl + '",'
            message = message + '"result":"' + result + '" }}'
            message = message + "\r\n"
        else:
            message = '''
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <style>
                body {
                color: black;
                font-family: verdana;
                }
                </style>
                <title>Your TensorShow</title>
            </head>
            <body>
            <section>
                <h1>Your image has been processed</h1>
            
                <p>Here is the sample picture behind the URL you gave</p>
                <img src="'''
            message = message + imgurl
                
            message = message + '''" alt="your sample image" width=280 height=280>
            <p>TensorFlow says that the number in that picture is:'''
            message = message + result
            message = message + '''</p>
            <p>Perhaps in the version 2.0 there would be a possibility for feedback the AI whether the prediction was right or not...</p>
            </section>
            </body>
            </html>

            '''
    else:
        message = '''<!doctype html>
        <html lang="en">
        <head>
            <style>
                body {
                color: black;
                font-family: verdana;
                }
            </style>
            <meta charset="utf-8">
            <title>Welcome to TensorShow</title>
        </head>
        <body>
        <section>
        <h1>Welcome to TensorShow!</h1>
        <h2>What is this?</h2>
        <p>This is my hobby/study project a TensorFlow based image recognition engine.
        It is based on the MNIST example and an excellent article at <a href="https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html">MNIST for ML Beginners</a>
        This implementation simply expands the sample code so that one can also feed own images to it.
        Those images will be transformed and resized to same 28x28 used by the MNIST example. 
        
        The result i.e. the prediction will be available in HTML page and JSON format</p>
        
        <h2>Can I try it?</h2>
        <p>If you want to give it a spin you can simply feed it any URL, but a clear picture of a figure would be ideal
        to make any sense to this. To call the engine you simply add <i>?imgurl=</i> as a parameter for this site.
        This one works well:</p>
        <p><code><a href="http://tensorshow.herokuapp.com/?imgurl=http://siutou.dlinkddns.com/site/img.jpg">
        http://tensorshow.herokuapp.com/?imgurl=http://siutou.dlinkddns.com/site/img.jpg</a></code></p>
        <p>Here is the sample picture behind that URL</p>
        <img src="http://siutou.dlinkddns.com/site/img.jpg" alt="img.jpg" width=280 height=280>
        <section>
        <section>
        <p>It is also possible to get result in JSON format. Simply add extra parameter <i>json=yes</i>
        for example like this:</p>
        <p><code><a href="http://tensorshow.herokuapp.com/?imgurl=http://siutou.dlinkddns.com/site/img.jpg&json=yes">
        http://tensorshow.herokuapp.com/?imgurl=http://siutou.dlinkddns.com/site/img.jpg&json=yes</a></code></p>
        </section>
        <section>
        <h2>The source code</h2>
        <p>The source code will be very shortly at <a href="https://github.com/MattiKankkonen">GitHub</a> any minute now...</p>
        </section>
        </body>
        </html>
        '''
    return message 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    train_tensor()
    app.run(host='0.0.0.0', port=port)

