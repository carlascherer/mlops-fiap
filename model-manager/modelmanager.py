import datetime
import json
import logging
import requests
from flask import Flask, request, Response

app = Flask(__name__)

logging.basicConfig(filename='/log/modelmanager.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def logapp(jsoncontent, sufix):
    log_filename = f"/log/log_{sufix}.json"
    with open(log_filename, 'w') as logfile:
        json.dump(jsoncontent, logfile)

@app.route('/predict',methods=['GET','POST'])
def predict(request = request):
    global SITE_NAME
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    reqtime = datetime.datetime.now()
    logg_track = {
        "reqtime": str(reqtime),
        "REMOTE_ADDR": request.environ.get('REMOTE_ADDR'),
        "input": {
            "base_url": request.base_url,
            "method": request.method,
            "args": request.args,
            "content": {}
        },
        "output": {}
    }

    with open('./config/microservices.json') as json_file:
        microservices_config = json.load(json_file)

    try:
        mymodel = request.args['model']
        mymodel_url = microservices_config["models"][mymodel]['url']
        logg_track["model"] = mymodel
    except KeyError:
        raise Exception("O modelo deve ser informado no argumento 'model' e deve ser um modelo válido nas configuracoes (config/microservices.json)")

    json_content = request.get_json(silent=True, force=True) or {}
    logg_track["input"]["content"] = json_content

    print(mymodel_url)
    print(json_content)

    if request.method=='GET':
        resp = requests.get(url=mymodel_url, json=json_content)
    elif request.method=='POST':
        resp = requests.post(url=mymodel_url, json=json_content)
    else:
        raise Exception("Método não suportado")

    headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    response = Response(resp.content, resp.status_code, headers)
    resp_content = json.loads(resp.content)
    logg_track["output"].update({
        "content": resp_content,
        "status_code": resp.status_code,
        "headers": headers
    })
    logapp(jsoncontent=logg_track, sufix=reqtime.strftime("%Y%m%d-%H%M%S.%f"))
    return response

if __name__ == '__main__':
    logging.info('Executando modelmanager...')

    app.run(host='0.0.0.0', port=8080)