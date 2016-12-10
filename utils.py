from datetime import datetime
from elasticsearch import Elasticsearch

def initDB():
    # elasticsearch
    dt = datetime.now()
    exp_id = "rnn-" + dt.strftime("%y-%m-%d-%H-%M")
    rnn_id = "rnn_weights"
    es = Elasticsearch()
    return (es, exp_id, rnn_id)

def sendDocDB(handle,doc):
    es, exp_id, _ = handle
    doc["timestamp"] = datetime.utcnow()
    doc["experiment"] = args.exp_id
    #doc["model"] = model.model.get_config()
    #print idx, doc
    es.index(index=exp_id, doc_type="textgame_result2", body=doc)

def sendModelDB(handle,model):
    es, _, _ = handle
    model_idx = "model"
    doc = vars(model)
    doc["experiment"] = args.exp_id
    es.index(index=model_idx, doc_type="model_config", body=doc)
