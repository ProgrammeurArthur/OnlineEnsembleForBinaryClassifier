import model as m
from river import  metrics as me,utils
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


def ensemble_predict_one( estimators,x, opcao,y):
    for  model in estimators:
        y_pred = model.get_model().predict_one(x)
        #print(y_pred)
        model.set_predict_ensemble(y_pred)
        if y_pred is not None:
            for metric in model.get_metrics().values():
                metric.update(y_true=y, y_pred=y_pred)

    match opcao:
        case '2':
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
        case '3':
            for model in estimators:
                print(f"{model.get_name()} - RollingAccuracy: {model.get_metrics()['RollingAccuracy'].get()}")
            #RollingAccuracy
            best_model = max(estimators, key=lambda model: model.get_metrics()["RollingAccuracy"].get())
            return best_model.get_predict_ensemble(), best_model.get_name()
        case '4':
            predictions=0.0
            # Método best model average
            predictions += sum(model.get_metrics()["RollingAccuracy"].get() for model in estimators)
            print(f'predicao total:{predictions}')
            average_prediction = predictions / len(estimators)
            print(f'predicao media:{average_prediction}')
            # Escolher apenas modelos cujas previsões estão acima da média
            selected_models = [model for model in estimators if model.get_metrics()["RollingAccuracy"].get() > average_prediction]
            print([model.get_name() for model in selected_models])
            return func_voting(selected_models)
        case '5':
            threshold = 0.9
            selected_models = [model for model in estimators if model.get_metrics()["RollingAccuracy"].get() > threshold]
            print([model.get_name() for model in selected_models])
            return func_voting(selected_models)
        case '6':
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
      
def ensemble_learn_one(estimators,x,y):
    for model in estimators:
        model.get_model().learn_one(x, y)


class Model_class:
    def __init__(self,name, model, metrics):
        self.name=name
        self.model=model
        self.metrics=metrics

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