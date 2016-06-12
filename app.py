'''
Created on 28 May 2016

@author: Matti Kankkonen
'''
import os
from flask import Flask
from flask import request
from flask import jsonify

import urllib
import numpy as np
import re
from PIL import Image, ImageDraw

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

debug = 0

app = Flask(__name__, static_url_path='/tmp/')

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
    print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    ###
    if(debug):
        row = col = 28
        data = np.zeros((1,row*col))
        print "Reverse verification"
        image_index = 6
        #print mnist.test.images[image_index]
        print mnist.test.labels[image_index]
        np.copyto(data, mnist.test.images[image_index])
        #print "Prediction ", prediction.eval(feed_dict={x: data}, session=sess)
        #print "Probabilities", probabilities.eval(feed_dict={x: data}, session=sess)

        for yy in range(row):
            row_str = []
            for xx in range(col):
                value = mnist.test.images[image_index,yy*col + xx]
            
                row_str.append(int(value*99))
                #im.putpixel((xx,yy), value)
            print row_str
        
        #im.save("/mnt/tensorshow/reference.jpg")
    
    
def crop_image(data, fileName):
    image = Image.open(fileName)
    col, row = image.size
    
    frame = 30;
    left = right = up = down = 0
    '''' 
    Ok, first let's find out where in the picture
    our figure is located 
    '''
    for i in range(3, row):
        for j in range(3, col):
            value = data[0,i*col + j]
            if (value != 0):
                if(left == 0 or left > j):
                    left = j
                if(right == 0 or right < j):
                    right = j
                if(up == 0 or up > i):
                    up = i
                if(down == 0 or down < i):
                    down = i
    
    ''' Not too tight crop, we want some frame around '''
    left = left - frame
    if(left < 0):
        left = 0
    right = right + frame
    if(right > col):
        right = col
    up = up - frame
    if(up < 0):
        up = 0
    down = down + frame
    if(down > row):
        down = row
    
    ''' Finally we need to ensure it's still a rectangle '''
    x = right - left
    y = down - up
    if ((x) > (y)):
        # it's wider than it's high
        adjust = int((x - y)/2)
        up = up - adjust
        down = down + adjust
    else:
        adjust = int((y - x)/2)
        left = left - adjust
        right = right + adjust
         
    sample = image.crop([left, up, right, down])
    ''' 
    If we cropped outside of original image we need to 
    fill the new areas 
    '''
    x1 = y1 = 0
    x2 = col
    y2 = row
    draw = False
    if(left < 0):
        x2 = abs(left)
        draw = True
    if(right > col):
        x2 = col + (right - col)
        x1 = col
        draw = True
    if(up < 0):
        y2 = abs(up)
        draw = True
    if(down > row):
        y2 = row + (down - row)
        y1 = row
        draw = True
    if(draw):
        print left, up, right, down
        print x1, y1, x2, y2
        d = ImageDraw.Draw(sample)
        d.rectangle([x1, y1, x2, y2], 255)
    
    sample.save(fileName)
                        
def prepare_image(fromFile, toSize, toFile):
    orig = Image.open(fromFile)
        
    im = orig.convert("L")
    im = im.resize(toSize)
    col, row = im.size
    data = np.zeros((1,row*col))
    pixels = im.load()
    
    for i in range(row):
        for j in range(col):
            value = pixels[j,i]
            data[0,i*col + j] = value
                
    for i in range(row):
        orig_row_str = []
        row_str = []
        for j in range(col):
            orig_row_str.append( data[0,i*col + j] )
            value = (255 - data[0,i*col + j])
            
            value = value / 250
            if (value > 1):
                value = 1
            if (value < 0.50):
                value = 0
            row_str.append(int(value*99))
            data[0,i*col + j] = value
        
        if(debug):
            #print orig_row_str
            print row_str
    
    im.save(toFile)
    return data
    
    
def evaluate_image(dir, fileName, crop):    
    if(crop == False):
        data = prepare_image(dir+fileName, [28,28], dir + "latest_sample.jpg")    
    else:
        data = prepare_image(dir+fileName, [280,280], dir+"crop.jpg")
        crop_image(data, dir + "crop.jpg")
        data = prepare_image(dir+"crop.jpg", [28,28], dir + "latest_sample.jpg")
        
    ''' The training is done earlier and now we should try to use 
    the model to recognise our picture. '''
    prediction = tf.argmax(y,1)
    probabilities = y
    
    #print "Probabilities", probabilities.eval(feed_dict={x: data}, session=sess)[0].astype(float)
    prob = probabilities.eval(feed_dict={x: data}, session=sess)[0].astype(float)
    evaluation = prediction.eval(feed_dict={x: data}, session=sess)[0].astype(int)
    #print "Probability " , prob[evaluation] 
    #return prediction.eval(feed_dict={x: data}, session=sess)[0].astype(str)
    return evaluation, prob[evaluation]
 
 
@app.after_request
def apply_allow_origin(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

@app.route("/env")
def list_env():
    response_body = ['%s: %s' % (key, value)for key, value in sorted(app.config.items())]
    response_body = '\n'.join(response_body)
    return response_body
        
@app.route("/latest_sample.jpg")
def send_latest():
    return app.send_static_file("latest_sample.jpg")


@app.route("/")
def hello():
    
    imgurl=request.args.get('imgurl')
    usejson=request.args.get('json')
    crop = request.args.get('crop')    
    if (imgurl != None):
        dir = "/tmp/"
        
        # TODO: urllib is handy but doesn't tell anything 
        # about success so currently if we get invalid URL 
        # we simply crash. 
        #
        # URL handling also lacking user input sanitation!!
        # Of course not to mention the concurrency support
        # or more like the lack of. It's just currently
        # very convenient to save the images to file for debugging
        urllib.urlretrieve(imgurl, dir+"img.jpg")
        result = "invalid"
        #prepare_image(dir)
        if (crop != None):
            result = evaluate_image(dir, "img.jpg", True)
        else:
            result = evaluate_image(dir, "img.jpg", False)        
        
        if (usejson != None):           
            jsonMsg = {'status':'invalid', 'url': '', 'result': '', 'probability':''}
            
            if (result != "invalid"):
                jsonMsg['status']='valid'
            jsonMsg['url']=imgurl
            jsonMsg['result']=result[0]
            jsonMsg['probability']=result[1]
            return jsonify(jsonMsg);
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
            message = message + str(result[0])
            message = message + " with probability of "
            message = message + str(result[1])
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

