import model as m
from river import datasets
from river import  metrics as me,utils
from river import ensemble as e, compose
from river import linear_model as lm
from river import preprocessing as pp
#rom river.base import MetaEstimatorMixin
from river.base import Classifier
from river import stats
import time as t
import aux as a
import numpy as np
metric_pref="Accuracy"
# def createList(models, metrics):
#     estimators=[]
#     for name, model in models.items():
#         estimators.append(Model_class(name, model, metrics))
#     return estimators
def createList(models, metrics):
    estimators = []
    for name, model in models.items():
        model_metrics = {metric_name: metric.clone() for metric_name, metric in metrics.items()}
        estimators.append(Model_class(name, model, model_metrics))
    return estimators


def ensemble_predict_one( estimators,x, opcao,y, dataset):
    for  model in estimators:
        model.round+=1
        start_time_predict = t.time()
        y_pred = model.get_model().predict_one(x)
        if isinstance(dataset, datasets.HTTP):
            y_pred= a.aux_HTTP(y_pred)
        elif isinstance(dataset, datasets.TREC07):
            y_pred=a.aux_TRE07(y_pred)

        end_time_predict = t.time()
        model.Time2predict=a.calTime(start_time_predict,end_time_predict)
        #print(y_pred)
        model.set_predict_ensemble(y_pred)
        if y_pred is not None:
            for metric in model.get_metrics().values():
                metric.update(y_true=y, y_pred=y_pred)

    #match opcao:
    #    case '2':
    if(opcao=='2'):

            # true_count =0
            # false_count=0
            # none_count=0
            # for model in estimators:
            #     aux= model.get_predict_ensemble()
            #     if aux== True:
            #         true_count+=1
            #     elif aux==False:
            #         false_count +=1
            #     else:
            #         none_count+=1
            # if true_count > false_count:
            #   return True
            # elif false_count > true_count:
            #     return False
        return func_voting(estimators)
        #case '3':
    elif(opcao=='3'):
            # for model in estimators:
            #     print(f"{model.get_name()} - RollingAccuracy: {model.get_metrics()['RollingAccuracy'].get()}")
            #RollingAccuracy
        best_model = max(estimators, key=lambda model: model.get_metrics()[metric_pref].get())
        return best_model.get_predict_ensemble(), best_model.get_name()
        #case '4':
    elif(opcao=='4'):

        predictions=0.0
        # Método best model average
        predictions += sum(model.get_metrics()[metric_pref].get() for model in estimators)
        #print(f'predicao total:{predictions}')
        average_prediction = predictions / len(estimators)
        #print(f'predicao media:{average_prediction}')
        # Escolher apenas modelos cujas previsões estão acima da média
        selected_models = [model for model in estimators if model.get_metrics()[metric_pref].get() > average_prediction]
        #print([model.get_name() for model in selected_models])
        return func_voting(selected_models)
        #case '5':
    elif(opcao=='5'):

        threshold = 0.8
        selected_models = [model for model in estimators if model.get_metrics()[metric_pref].get() > threshold]
        #print([model.get_name() for model in selected_models])
        return func_voting(selected_models)
        #case '6':
    else:
        pass

def func_voting(estimators):
    true_count =0
    false_count=0
    none_count=0
    for model in estimators:
        aux= model.get_predict_ensemble()
        if aux== True:
            true_count+=1
        elif aux==False:
            false_count +=1
        else:
            none_count+=1
    if true_count >= false_count:
        return True
    elif false_count > true_count:
        return False

def criar_Csv_models(estimators, name_bd, name_ensemble):
    for model in estimators:
        a.criarCSV(model.buffer,model.name, name_bd, name_ensemble)


def ensemble_learn_one(estimators,x,y):
    for model in estimators:
        
        start_time_learn = t.time()
        model.get_model().learn_one(x, y)
        end_time_learn = t.time()
        model.Time2learn=a.calTime(start_time_learn, end_time_learn)
        model.Timestamp+=( model.Time2learn + model.Time2predict)
        name=model.get_name()
        metrics_data=a.dados(model.round,model.Timestamp, model.Time2predict, model.Time2learn, model.metrics, name)
        model.buffer.append(metrics_data)

class Model_class:
    def __init__(self,name, model, metrics):
        self.name=name
        self.model=model
        self.metrics=metrics
        self.buffer=[]
        self.round=0
        self.Timestamp=0.0
        self.Time2learn=0.0
        self.Time2predict=0.0

    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def get_metrics(self):
        return self.metrics
    
    def set_predict_ensemble(self,y_pred):
        self.y_pred_ensemble=y_pred

    def get_predict_ensemble(self):
        return self.y_pred_ensemble


