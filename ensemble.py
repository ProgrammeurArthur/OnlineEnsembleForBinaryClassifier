import model as m
from river import  metrics as me,utils
def createList(models, metrics):
    estimators=[]
    for name, model in models.items():
        estimators.append(Model_class(name, model, metrics))
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
            if true_count > false_count:
              return True
            elif false_count > true_count:
                return False
        case '3':
            pass
        case '4':
            pass
        case '5':
            pass
        case '6':
            pass

      
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