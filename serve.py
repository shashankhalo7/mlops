import logging, requests, os, io, glob, time
from fastai.vision import *
import pandas as pd
import wget
from string import ascii_letters
import urllib.parse
import time
import base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json' 
JPEG_CONTENT_TYPE = 'image/jpeg'

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    torch.no_grad()
    logger.info('model_fn')
    path = Path(model_dir)
    learn=load_learner(path,file='model.pkl')
    learn.to_fp32()
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    torch.no_grad()
    logger.info('Deserializing the input data.')
    if content_type == JPEG_CONTENT_TYPE: return [open_image(io.BytesIO(base64.b64decode(x.encode('utf-8')))) for x in json.loads(request_body)['img_data']]
    # process a URL submitted to the endpoint
    #print("Input Started No JPEG")
    if content_type == JSON_CONTENT_TYPE:
        request=json.loads(request_body)
        img_list=[]
        global urls
        urls=request['url']
        for i in request['url']:
            img_request = requests.get(i, stream=True)
            img_list.append(open_image(io.BytesIO(img_request.content)))
        return img_list 
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, learner):
    model=learner.model
    np.random.seed(42)
    tfms = None
    if  (os.path.exists("test"))==False:
        os.mkdir("test")
    bs=len(input_object)
    for i,img in enumerate(input_object):
        img.save("test/{0:0=2d}.jpg".format(i))
    now=time.time()
    #print(f"Saving temp images data time:{now-then}")
    test = ImageList.from_folder(path='test')
    learner.data.add_test(test)
    preds=learner.get_preds(ds_type=DatasetType.Test)
    classes=[learner.data.classes[x] for x in np.argmax(preds[0],1)]
    output={int((str(learner.data.test_ds.items[i]).strip(ascii_letters+'/'+'.'+'_'))):classes[i] for i in range(bs)}
    for i in os.listdir("test/"):
        os.remove(f"test/{i}")
    return dict(urls=urls,predictions=[output[i] for i in range(bs)],Status='Success')

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    print("Output started")
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))


import flask
from flask import Flask, Response
import pandas as pd
from io import StringIO
app = Flask(__name__)

learner=model_fn('/opt/ml/model/')

@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    """
    # we will return status ok if the model doesn't barf
    # but you can also insert slightly more sophisticated tests here
    try:
        print("Ping Received")
        return Response(response='{"status": "ok"}', status=200, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def predict():
    request_body = flask.request.data
    print(request_body,type(request_body))
    content_type= json.loads(request_body).get('content_type','application/json') 
    input_object=input_fn(request_body, content_type=content_type)
    prediction=predict_fn(input_object, learner)
   
    # format into a csv
    output=output_fn(prediction)[0]
    
    # return
    return Response(response=output, status=200, mimetype='application.json')
