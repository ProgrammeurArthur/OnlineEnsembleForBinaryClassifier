from river import datasets, metrics, utils
import pandas as pd
window_size=100
def escolha_BD():
    opcao= input("Digite o banco de dados que deseja:\n1-SmsSpam\n2-Bananas\n3-Elec2\n4-HTTP\n5-MaliciousURL(verificar)\n6-Phishing\n7-SMTP\n8-TREC07(verificar)\n9-Higgs\n")
    #databases
    #dataset=datasets.SMSSpam()
    #dataset_Bananas=datasets.Bananas()
    #dataset_CreditCard=datasets.CreditCard()
    #dataset_Elec2=datasets.Elec2()
    #dataset_HTTP=datasets.HTTP()
    #dataset_MaliciousURL=datasets.MaliciousURL()
    #dataset_Phishing=datasets.Phishing()
    #dataset_SMTP=datasets.SMTP()
    #dataset_TREC07=datasets.TREC07()
    #dataset_Higgs=datasets.Higgs()
    match opcao:
        case '1':
            dataset=datasets.SMSSpam()
            name='SMSSpam'
        case '2':
            dataset=datasets.Bananas()
            name='Bananas'
        case '3':
            dataset=datasets.CreditCard()
            name='CreditCard'
        case '4':
            dataset=datasets.Elec2()
            name='Elec2'
        case '5':
            dataset=datasets.MaliciousURL()
            name='MaliciousURL'
        case '6':
            dataset=datasets.Phishing()
            name='Phishing'
        case '7':
            dataset=datasets.SMTP()
            name='SMTP'
        case '8':
            dataset=datasets.TREC07()
            name='TREC07'
        case '9':
            dataset=datasets.Higgs()
            name='Higgs'
        case _:
            print("[ERRO]Opção inválida.")
    return dataset, name


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
    "RollingROCAUC": metrics.RollingROCAUC(window_size=4),
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


def calTime(tempo1, tempo2):
    return tempo2 - tempo1

def dados(Round, Timestamp, Time2predict, Time2learn, metrics_dict, algoritmo):
    metrics_data = {'Round': Round, 
                       'Timestamp': Timestamp, 
                       'Time2predict': Time2predict, 
                       'Time2learn': Time2learn
    }
        
    for metric_name, metric in metrics_dict.items():
        metrics_data[metric_name] = metric.get()

    metrics_data['Algorithm']= algoritmo


    return metrics_data


def criarCSV(buffer, name_model, name_bd):
    df = pd.DataFrame(buffer)
    csv_file = f'{name_model}_{name_bd}_resultados.csv'
    df.to_csv(csv_file, index=False)
    


