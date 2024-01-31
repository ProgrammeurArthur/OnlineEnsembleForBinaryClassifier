from river import datasets, metrics, utils
import model as m
import time as t
import aux as a
window_size=100
import ensemble as e 
#from river import ensemble
from river.linear_model import LogisticRegression as lr
from river.tree import HoeffdingTreeClassifier as htc
from tqdm import tqdm

dataset, name_bd= a.escolha_BD()

metrics_dict = {
    "Accuracy": metrics.Accuracy(),
    "ROCAUC": metrics.ROCAUC(),
    "BalancedAccuracy": metrics.BalancedAccuracy(),
    "CohenKappa": metrics.CohenKappa(),
    "Completeness": metrics.Completeness(),
    "F1": metrics.F1(),
    "FBeta": metrics.FBeta(beta=2),
    "FowlkesMallows": metrics.FowlkesMallows(),
    "GeometricMean": metrics.GeometricMean(),
    "LogLoss": metrics.LogLoss(),
    "MCC": metrics.MCC(),
    "Precision": metrics.Precision(),
    "WeightedPrecision": metrics.WeightedPrecision(),
    "WeightedRecall": metrics.WeightedRecall(),
    #rolling
    "RollingAccuracy" : utils.Rolling(metrics.Accuracy(), window_size),
    "RollingROCAUC": metrics.RollingROCAUC(window_size),
    "RollingBalancedAccuracy" : utils.Rolling(metrics.BalancedAccuracy(), window_size),
    "RollingCohenKappa" : utils.Rolling(metrics.CohenKappa(), window_size),
    "RollingCompleteness" : utils.Rolling(metrics.Completeness(), window_size),
    "RollingF1" : utils.Rolling(metrics.F1(), window_size),
    "RollingFBeta" : utils.Rolling(metrics.FBeta(beta=2), window_size),
    "RollingFowlkesMallows" : utils.Rolling(metrics.FowlkesMallows(), window_size),
    "RollingGeometricMean" : utils.Rolling(metrics.GeometricMean(), window_size),
    "RollingLogLoss" : utils.Rolling(metrics.LogLoss(), window_size),
    "RollingMCC" : utils.Rolling(metrics.MCC(), window_size),
    "RollingPrecision" : utils.Rolling(metrics.Precision(), window_size),
    "RollingWeightedPrecision" : utils.Rolling(metrics.WeightedPrecision(), window_size),
    "RollingWeightedRecall" : utils.Rolling(metrics.WeightedRecall(), window_size)

}

model_B= m.BernoulliNB(dataset)
model_ARF= m.ARFClassifier(dataset)
model_Hard= m.HardSamplingClassifier(dataset)
model_ROS= m.RandomOverSampler(dataset)
model_RS= m.RandomSampler(dataset)
model_RUS= m.RandomUnderSampler(dataset)
model_LR= m.LogisticRegression(dataset)
model_P= m.Perceptron(dataset)
model_OVOC= m.OneVsOneClassifier(dataset)
model_OVRC= m.OneVsRestClassifier(dataset)
model_OCC= m.OutputCodeClassifier(dataset)
model_EFDTC= m.tree_ExtremelyFastDecisionTreeClassifier(dataset)
model_HATC= m.tree_HoeffdingAdaptiveTreeClassifier(dataset)
model_HTC= m.tree_HoeffdingTreeClassifier(dataset)
#model_SGTC= m.tree_SGTClassifier(dataset)
model_ABC= m.ensemble_ADWINBaggingClassifier(dataset)
model_ABOC= m.ensemble_ADWINBoostingClassifier(dataset)
model_BC= m.ensemble_BOLEClassifier(dataset)
model_ADABC= m.ensemble_AdaBoostClassifier(dataset)
model_BAC= m.ensemble_BaggingClassifier(dataset)
model_LBC= m.ensemble_LeveragingBaggingClassifier(dataset)
model_SRPC= m.ensemble_SRPClassifier(dataset)
model_SC= m.ensemble_StackingClassifier(dataset)
model_VC= m.ensemble_VotingClassifier(dataset)
model_AMFC= m.forest_AMFClassifier(dataset)
model_ALMA=m.linear_model_ALMAClassifier(dataset)
model_PAC=m.linear_model_PAClassifier(dataset)
model_SoftmaxRegression=m.linear_model_SoftmaxRegression(dataset)
model_ComplementNB=m.naive_bayes_ComplementNB(dataset)
model_GaussianNB=m.naive_bayes_GaussianNB(dataset)
model_MultinomialNB=m.naive_bayes_MultinomialNB(dataset)
model_KNNClassifier=m.neighbors_KNNClassifier(dataset)

models={
     "BernoulliNB": model_B,
    "ARFClassifier": model_ARF,
    "HardSamplingClassifier": model_Hard,
    "RandomOverSampler": model_ROS,
    "RandomSampler": model_RS,
    "RandomUnderSampler": model_RUS,
    "LogisticRegression": model_LR,
    "Perceptron": model_P,
    "OneVsOneClassifier": model_OVOC,
    "OneVsRestClassifier": model_OVRC,
    "OutputCodeClassifier": model_OCC, 
    "tree_ExtremelyFastDecisionTreeClassifier": model_EFDTC,
    "tree_HoeffdingAdaptiveTreeClassifier": model_HATC,
    "tree_HoeffdingTreeClassifier": model_HTC,
    #"tree_SGTClassifier": model_SGTC,
    "ensemble_ADWINBaggingClassifier": model_ABC,
    "ensemble_ADWINBoostingClassifier": model_ABOC,
    "ensemble_BOLEClassifier": model_BC,
    "ensemble_AdaBoostClassifier": model_ADABC,
    "ensemble_BaggingClassifier": model_BAC,
    "ensemble_LeveragingBaggingClassifier": model_LBC,
    "ensemble_SRPClassifier": model_SRPC,
    "ensemble_StackingClassifier": model_SC,
    "ensemble_VotingClassifier": model_VC,
    "forest_AMFClassifier": model_AMFC,
    "model_linear_ALMAClassifier":model_ALMA,
    "linear_model_PAClassifier":model_PAC,
    "linear_model_SoftmaxRegression":model_SoftmaxRegression,
    #"model_ComplementNB":model_ComplementNB,
    "model_GaussianNB":model_GaussianNB,
    #"model_MultinomialNB":model_MultinomialNB,
    "model_KNNClassifier":model_KNNClassifier

}
while(1):
    opcao= input("Digite a opcao que deseja criar:\n1-Classificadores indepedentes\n2-Ensemble Voting\n3-Ensemble BM\n4-Ensemble BWM\n5-threshold\n")
    if(opcao=='1'):
        #for model in models.values():
        for model_name, model in models.items():
            buffer=[]
            metrics_dict=metrics_dict
            Round, Timestamp, Time2predict, Time2learn=0,0,0,0
            #print(model_name)
        
            t_start_time_Timestamp=t.time()
            for x, y in tqdm(dataset):

                start_time_Timestamp=t.time()
                Round+=1
                
                start_time_predict = t.time()
                y_pred = model.predict_one(x)
                if isinstance(dataset, datasets.HTTP):
                    y_pred= a.aux_HTTP(y_pred)
                print(f'y_pred:{y_pred}')
                print(model_name)

                end_time_predict = t.time()
                Time2predict= a.calTime(start_time_predict, end_time_predict)

                if y_pred is not None:
                    for metric in metrics_dict.values():
                        metric.update(y_true=y, y_pred=y_pred)

                start_time_learn = t.time()
                model.learn_one(x, y)
                end_time_learn = t.time()
                Time2learn= a.calTime(start_time_learn, end_time_learn)

                #end_time_Timestamp=t.time()
                Timestamp+= (Time2learn+Time2predict)

                metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict,model_name)
                buffer.append(metrics_data)
                #metrics_data = {'Round': Round, 
                #               'Timestamp': Timestamp, 
                #               'Time2predict': Time2predict, 
                #               'Time2learn': Time2learn
                #               }
                #buffer.append(metrics_data)
                #print(buffer)
                #for metric_name, metric in metrics_dict.items():
                #    metrics_data[metric_name] = metric.get()

                #buffer.append(metrics_data)
                #print(metrics_data)


            #t_end_time_Timestamp=t.time()
            #Timestamp= a.calTime(t_start_time_Timestamp,t_end_time_Timestamp)
            #Round+=1
            #metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, model_name)
            #buffer.append(metrics_data)
            #print(buffer)
            a.criarCSV(buffer,model_name, name_bd, model_name)
            
        # result_dict = {metric_name: metric.get() for metric_name, metric in metrics_dict.items()}
        # print(result_dict)
    elif(opcao=='2'):
        buffer=[]
        metrics_dict=metrics_dict
        Round, Timestamp, Time2predict, Time2learn=0,0,0,0
        estimators=e.createList(models, metrics_dict)
        model_name='ensemble_voting'
        #print(estimators)
        Round=0
        #t_start_time_Timestamp=t.time()
        for x, y in tqdm(dataset):
             start_time_Timestamp=t.time()
             Round+=1 
             start_time_predict = t.time()
             estimators_y_pred=e.ensemble_predict_one(estimators,x,opcao,y, dataset)
             #print(estimators_y_pred)
             end_time_predict = t.time()
             Time2predict= a.calTime(start_time_predict, end_time_predict)

             if estimators_y_pred is not None:
                for metric in metrics_dict.values():
                    metric.update(y_true=y, y_pred=estimators_y_pred)

             start_time_learn = t.time()
             e.ensemble_learn_one(estimators, x, y)
             end_time_learn = t.time()
             Time2learn= a.calTime(start_time_learn, end_time_learn)
            

             #end_time_Timestamp=t.time()
             Timestamp+= (Time2learn+Time2predict)

             metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict,model_name)
             buffer.append(metrics_data)

        
        #t_end_time_Timestamp=t.time()
        #Timestamp= a.calTime(t_start_time_Timestamp,t_end_time_Timestamp)
        #Round+=1
        #metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, model_name)
        #buffer.append(metrics_data)
        #print(buffer)
        a.criarCSV(buffer,model_name, name_bd,model_name)
        e.criar_Csv_models(estimators, name_bd, model_name)
    elif(opcao=='3'):
        buffer=[]
        #metrics_dict=metrics_dict
        Round, Timestamp, Time2predict, Time2learn=0,0,0,0
        estimators=e.createList(models, metrics_dict)
        model_name='ensemble_best_model'
        ensemble_model_name='ensemble_best_model'
        #print(estimators)
        Round=0
        #t_start_time_Timestamp=t.time()
        for x, y in tqdm(dataset):
             start_time_Timestamp=t.time()
             Round+=1 
             start_time_predict = t.time()
             estimators_y_pred, model_name =e.ensemble_predict_one(estimators,x,opcao,y, dataset)
             #print(estimators_y_pred)
             end_time_predict = t.time()
             Time2predict= a.calTime(start_time_predict, end_time_predict)

             if estimators_y_pred is not None:
                for metric in metrics_dict.values():
                    metric.update(y_true=y, y_pred=estimators_y_pred)

             start_time_learn = t.time()
             e.ensemble_learn_one(estimators, x, y)
             end_time_learn = t.time()
             Time2learn= a.calTime(start_time_learn, end_time_learn)
            

             #end_time_Timestamp=t.time()
             Timestamp+= (Time2learn+Time2predict)

             metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict,model_name)
             buffer.append(metrics_data)

        
        #t_end_time_Timestamp=t.time()
        #Timestamp= a.calTime(t_start_time_Timestamp,t_end_time_Timestamp)
        #Round+=1
        #metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, model_name)
        #buffer.append(metrics_data)
        #print(buffer)
        a.criarCSV(buffer,ensemble_model_name, name_bd, ensemble_model_name)
        e.criar_Csv_models(estimators, name_bd, ensemble_model_name)

    elif(opcao=='4'):
        buffer=[]
        #metrics_dict=metrics_dict
        Round, Timestamp, Time2predict, Time2learn=0,0,0,0
        estimators=e.createList(models, metrics_dict)
        model_name='ensemble_best_model_average'
        ensemble_model_name='ensemble_best_model_average'
        #print(estimators)
        Round=0
        #t_start_time_Timestamp=t.time()
        for x, y in tqdm(dataset):
             #start_time_Timestamp=t.time()
             Round+=1 
             start_time_predict = t.time()
             estimators_y_pred =e.ensemble_predict_one(estimators,x,opcao,y, dataset)
             #print(estimators_y_pred)
             end_time_predict = t.time()
             Time2predict= a.calTime(start_time_predict, end_time_predict)

             if estimators_y_pred is not None:
                for metric in metrics_dict.values():
                    metric.update(y_true=y, y_pred=estimators_y_pred)

             start_time_learn = t.time()
             e.ensemble_learn_one(estimators, x, y)
             end_time_learn = t.time()
             Time2learn= a.calTime(start_time_learn, end_time_learn)
            

             end_time_Timestamp=t.time()
             Timestamp+= (Time2learn+Time2predict)

             metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict,model_name)
             buffer.append(metrics_data)

        
        #t_end_time_Timestamp=t.time()
        #Timestamp= a.calTime(t_start_time_Timestamp,t_end_time_Timestamp)
        #Round+=1
        #metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, model_name)
        #buffer.append(metrics_data)
        #print(buffer)
        a.criarCSV(buffer,ensemble_model_name, name_bd, ensemble_model_name)
        e.criar_Csv_models(estimators, name_bd, ensemble_model_name)

    elif(opcao=='5'):
        buffer=[]
        #metrics_dict=metrics_dict
        Round, Timestamp, Time2predict, Time2learn=0,0,0,0
        estimators=e.createList(models, metrics_dict)
        model_name='ensemble_best_model_threshold'
        ensemble_model_name='ensemble_best_model_threshold'
        #print(estimators)
        Round=0
        #t_start_time_Timestamp=t.time()
        for x, y in tqdm(dataset):
             #start_time_Timestamp=t.time()
             Round+=1 
             start_time_predict = t.time()
             estimators_y_pred =e.ensemble_predict_one(estimators,x,opcao,y, dataset)
             #print(estimators_y_pred)
             end_time_predict = t.time()
             Time2predict= a.calTime(start_time_predict, end_time_predict)

             if estimators_y_pred is not None:
                for metric in metrics_dict.values():
                    metric.update(y_true=y, y_pred=estimators_y_pred)

             start_time_learn = t.time()
             e.ensemble_learn_one(estimators, x, y)
             end_time_learn = t.time()
             Time2learn= a.calTime(start_time_learn, end_time_learn)
            

             #end_time_Timestamp=t.time()
             Timestamp+= (Time2learn+Time2predict)

             metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict,model_name)
             buffer.append(metrics_data)

        
        #t_end_time_Timestamp=t.time()
        #Timestamp= a.calTime(t_start_time_Timestamp,t_end_time_Timestamp)
        #Round+=1
        #metrics_data= a.dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, model_name)
        #buffer.append(metrics_data)
        #print(buffer)
        a.criarCSV(buffer,ensemble_model_name, name_bd, ensemble_model_name)
        e.criar_Csv_models(estimators, name_bd, ensemble_model_name)

    else:
        print('[ERRO] Opcao invalida!')
            