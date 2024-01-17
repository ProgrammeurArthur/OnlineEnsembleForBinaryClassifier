from river import datasets, evaluate, forest, metrics, stream, cluster, feature_extraction, naive_bayes,  imblearn, linear_model, optim, preprocessing
from river import multiclass, multioutput, tree, neural_net as nn, utils
from river import bandit
from river import proba
from river import stats, compat, compose,ensemble
from river import linear_model,drift, feature_selection,neighbors
import functools
from river import stream

from pprint import pprint

def model_database(dataset):
    if isinstance(dataset, datasets.SMSSpam):
        model =  compose.Select('body')
    elif isinstance(dataset, datasets.Bananas):
        model =  compose.Select('1', '2')
    elif isinstance(dataset, datasets.CreditCard):
        model =  compose.Select('Amount','Time','V1', 'V2','V3','V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12','V13','V14','V15','V16','V17','V18','V19', 'V20','V21', 'V22','V23','V24','V25','V26','V27','V28')
    elif isinstance(dataset, datasets.Elec2):
        model =  compose.Select('nswdemand','nswprice','period','transfer','vicdemand','vicprice')
    elif isinstance(dataset, datasets.HTTP):
        model =  compose.Select('dst_bytes','duration','src_bytes')
    elif isinstance(dataset, datasets.Higgs):
        model =  compose.Select('jet 1 b-tag', 'jet 1 eta','jet 1 phi','jet 1 pt','jet 2 b-tag','jet 2 eta','jet 2 phi','jet 2 pt','jet 3 b-tag','jet 3 eta','jet 3 phi','jet 3 pt','jet 4 b-tag','jet 4 eta','jet 4 phi','jet 4 pt','lepton eta','lepton pT','lepton phi', 'm_bb','m_jj','m_jjj', 'm_jlv', 'm_lv','m_wbb''m_wwbb','missing energy magnitude','missing energy phi')
    elif isinstance(dataset, datasets.MaliciousURL):
       #verificar
       pass
    elif isinstance(dataset, datasets.Phishing):
        model =  compose.Select('age_of_domain','anchor_from_other_domain','empty_server_form_handler','https','ip_in_url','is_popular','long_url','popup_window','request_from_other_domain')
    elif isinstance(dataset, datasets.SMTP):
        model =  compose.Select('dst_bytes','duration','src_bytes')
    elif isinstance(dataset, datasets.TREC07):
        #verificar
        model =   compose.Select('body')
        #,'date','recipients','sender','subject'
    return model



def BernoulliNB(dataset):
    if isinstance(dataset, datasets.SMSSpam):
        model =  model_database(dataset)
        model |= (
        feature_extraction.TFIDF(on='body') |
        naive_bayes.BernoulliNB(alpha=0)
        )
    elif isinstance(dataset, datasets.TREC07):
        #model =  model_database(dataset)
        model = (
          #feature_extraction.TFIDF(on='body') |
          #preprocessing.StandardScaler() |            
          naive_bayes.BernoulliNB(alpha=0)
        )
    elif isinstance(dataset, datasets.MaliciousURL):
        #model =  model_database(dataset)
        model = (
        preprocessing.StandardScaler() |
        naive_bayes.BernoulliNB(alpha=0)
        )
    else:
        model =  model_database(dataset)
        model |= (
        preprocessing.StandardScaler() |
        naive_bayes.BernoulliNB(alpha=0)
        )
    return model

def ARFClassifier(databases):
    model =  model_database(databases)
    model|= forest.ARFClassifier(seed=8, leaf_prediction="mc")
    return model

def HardSamplingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= (
        feature_extraction.TFIDF(on='body') |
        imblearn.HardSamplingClassifier(
            naive_bayes.BernoulliNB(alpha=0),
            p=0.1,
            size=40,
            seed=42,
        )
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= (
        #feature_extraction.TFIDF(on='body') |
        preprocessing.StandardScaler() |
        imblearn.HardSamplingClassifier(
            naive_bayes.BernoulliNB(alpha=0),
            p=0.1,
            size=40,
            seed=42,
        )
        )
    return model

def RandomOverSampler(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= imblearn.RandomOverSampler(
        (
            feature_extraction.TFIDF(on='body') |
            naive_bayes.BernoulliNB(alpha=0)
        ),
        desired_dist={False: 0.4, True: 0.6},
        seed=42
    )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= imblearn.RandomOverSampler(
        (
            preprocessing.StandardScaler() |
            naive_bayes.BernoulliNB(alpha=0)
        ),
        desired_dist={False: 0.4, True: 0.6},
        seed=42
    )
    return model

def RandomSampler(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= imblearn.RandomSampler(
        (
            feature_extraction.TFIDF(on='body') |
            naive_bayes.BernoulliNB(alpha=0)
        ),
        desired_dist={False: 0.4, True: 0.6},
        sampling_rate=0.8,
        seed=42
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= imblearn.RandomSampler(
        (
            preprocessing.StandardScaler() |
            naive_bayes.BernoulliNB(alpha=0)
        ),
        desired_dist={False: 0.4, True: 0.6},
        sampling_rate=0.8,
        seed=42
        )
    return model

def RandomUnderSampler(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= imblearn.RandomUnderSampler(
            (
                feature_extraction.TFIDF(on='body') |
                naive_bayes.BernoulliNB(alpha=0)
            ),
            desired_dist={False: 0.4, True: 0.6},
            seed=42
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= imblearn.RandomUnderSampler(
            (
                preprocessing.StandardScaler() |
                naive_bayes.BernoulliNB(alpha=0)
            ),
            desired_dist={False: 0.4, True: 0.6},
            seed=42
        )
    return model

def LogisticRegression(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= (
            feature_extraction.TFIDF(on='body') |
            linear_model.LogisticRegression(optimizer=optim.SGD(.1))
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression(optimizer=optim.SGD(.1))
        )
    return model

def Perceptron(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= (feature_extraction.TFIDF(on='body')  | linear_model.Perceptron())

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= ( preprocessing.StandardScaler() | linear_model.Perceptron())

    return model

def OneVsOneClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model|= (feature_extraction.TFIDF(on='body')  |multiclass.OneVsOneClassifier(linear_model.LogisticRegression()))

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= ( preprocessing.StandardScaler()  |multiclass.OneVsOneClassifier(linear_model.LogisticRegression()))

    return model

def OneVsRestClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model|= (feature_extraction.TFIDF(on='body')  |multiclass.OneVsRestClassifier(linear_model.LogisticRegression()))

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= ( preprocessing.StandardScaler()   |multiclass.OneVsRestClassifier(linear_model.LogisticRegression()))

    return model

def OutputCodeClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model|= (feature_extraction.TFIDF(on='body')  | multiclass.OutputCodeClassifier(
        classifier=linear_model.LogisticRegression(),
        code_size=10,
        coding_method='random',
        seed=1
        ))

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= ( preprocessing.StandardScaler()   | multiclass.OutputCodeClassifier(
        classifier=linear_model.LogisticRegression(),
        code_size=10,
        coding_method='random',
        seed=1
        ))

    return model

def tree_ExtremelyFastDecisionTreeClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= tree.ExtremelyFastDecisionTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='nba',
        #nominal_attributes=['elevel', 'car', 'zipcode'],
        min_samples_reevaluate=100
    )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model|=preprocessing.StandardScaler()
        model |= tree.ExtremelyFastDecisionTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='nba',
        #nominal_attributes=['elevel', 'car', 'zipcode'],
        min_samples_reevaluate=100
    )


    return model

def tree_HoeffdingAdaptiveTreeClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10,
        seed=0
        )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model|=preprocessing.StandardScaler()
        model |= tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10,
        seed=0
        )

    return model

def tree_HoeffdingTreeClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= tree.HoeffdingTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10
    )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model|=preprocessing.StandardScaler()
        model |= tree.HoeffdingTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10
    )

    return model

def tree_SGTClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= tree.SGTClassifier(
        feature_quantizer=tree.splitter.StaticQuantizer(
            n_bins=32, warm_start=10
        )
        )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model|=preprocessing.StandardScaler()
        model |= tree.SGTClassifier(
        feature_quantizer=tree.splitter.StaticQuantizer(
            n_bins=32, warm_start=10
        )
        )

    return model

def ensemble_ADWINBaggingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):

        model |= ensemble.ADWINBaggingClassifier(
        model=(
        feature_extraction.TFIDF(on='body')  |
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
        )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        #model|=preprocessing.StandardScaler()
        model |= ensemble.ADWINBaggingClassifier(
        model=(
        #feature_extraction.TFIDF(on='body')  |
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
        )
    return model

def ensemble_ADWINBoostingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= ensemble.ADWINBoostingClassifier(
        model=(
            feature_extraction.TFIDF(on='body')  |
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
    )

    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        #model|=preprocessing.StandardScaler()
        model |= ensemble.ADWINBoostingClassifier(
        model=(
            #feature_extraction.TFIDF(on='body')  |
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
    )
    return model

def ensemble_AdaBoostClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= ensemble.AdaBoostClassifier(
        model=(
            tree.HoeffdingTreeClassifier(
                split_criterion='gini',
                delta=1e-5,
                grace_period=2000
            )
        ),
        n_models=5,
        seed=42
    )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= preprocessing.StandardScaler()
        model |= ensemble.AdaBoostClassifier(
        model=(
            tree.HoeffdingTreeClassifier(
                split_criterion='gini',
                delta=1e-5,
                grace_period=2000
            )
        ),
        n_models=5,
        seed=42
        )
    return model

def ensemble_BOLEClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= ensemble.BOLEClassifier(
        model=drift.DriftRetrainingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            drift_detector=drift.binary.DDM()
        ),
        n_models=10,
        seed=42
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= preprocessing.StandardScaler()
        model |= ensemble.BOLEClassifier(
        model=drift.DriftRetrainingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            drift_detector=drift.binary.DDM()
        ),
        n_models=10,
        seed=42
        )
    return model

def ensemble_BaggingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model = ensemble.BaggingClassifier(
        model=(
            feature_extraction.TFIDF(on='body')  |
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= preprocessing.StandardScaler()
        model = ensemble.BaggingClassifier(
        model=(
            #feature_extraction.TFIDF(on='body')  |
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
        )
    return model

def ensemble_LeveragingBaggingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model = ensemble.LeveragingBaggingClassifier(
            model=(
                feature_extraction.TFIDF(on='body')  |
                preprocessing.StandardScaler() |
                linear_model.LogisticRegression()
            ),
            n_models=3,
            seed=42
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= preprocessing.StandardScaler()
        model = ensemble.LeveragingBaggingClassifier(
            model=(
                #feature_extraction.TFIDF(on='body')  |
                preprocessing.StandardScaler() |
                linear_model.LogisticRegression()
            ),
            n_models=3,
            seed=42
        )
    return model

def ensemble_SRPClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        base_model = tree.HoeffdingTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10
        )
        model = ensemble.SRPClassifier(
            model=base_model, n_models=3, seed=42,
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        base_model = tree.HoeffdingTreeClassifier(
        grace_period=100,
        split_criterion='info_gain',
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10
        )
        model = ensemble.SRPClassifier(
            model=base_model, n_models=3, seed=42,
        )
    return model

def ensemble_StackingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= compose.Pipeline(
        ('feature',feature_extraction.TFIDF(on='body')),
        ('scale', preprocessing.StandardScaler()),
        ('stack', ensemble.StackingClassifier(
            [
                linear_model.LogisticRegression(),
                linear_model.PAClassifier(mode=1, C=0.01),
                linear_model.PAClassifier(mode=2, C=0.01),
            ],
            meta_classifier=linear_model.LogisticRegression()
        ))
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= compose.Pipeline(
        #(#'feature',feature_extraction.TFIDF(on='body')),
        ('scale', preprocessing.StandardScaler()),
        ('stack', ensemble.StackingClassifier(
            [
                linear_model.LogisticRegression(),
                linear_model.PAClassifier(mode=1, C=0.01),
                linear_model.PAClassifier(mode=2, C=0.01),
            ],
            meta_classifier=linear_model.LogisticRegression()
        ))
        )
    return model

def ensemble_VotingClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= (
            feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            ensemble.VotingClassifier([
                linear_model.LogisticRegression(),
                tree.HoeffdingTreeClassifier(),
                naive_bayes.GaussianNB()
            ])
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= (
            #feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            ensemble.VotingClassifier([
                linear_model.LogisticRegression(),
                tree.HoeffdingTreeClassifier(),
                naive_bayes.GaussianNB()
            ])
        )
    return model

def forest_AMFClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model|=(feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            forest.AMFClassifier(
            n_estimators=10,
            use_aggregation=True,
            dirichlet=0.5,
            seed=1
        ))
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model|=(#feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            forest.AMFClassifier(
            n_estimators=10,
            use_aggregation=True,
            dirichlet=0.5,
            seed=1
        ))
    return model

def linear_model_ALMAClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model |= (
        feature_extraction.TFIDF(on='body')|
        preprocessing.StandardScaler() |
        linear_model.ALMAClassifier()
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= (
        #feature_extraction.TFIDF(on='body')|
        preprocessing.StandardScaler() |
        linear_model.ALMAClassifier()
        )
    return model

def linear_model_PAClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):
        model = (
            feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            linear_model.PAClassifier(
            C=0.01,
            mode=1
        ))
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model = (
            #feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            linear_model.PAClassifier(
            C=0.01,
            mode=1
        ))
    return model

def linear_model_SoftmaxRegression(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):               
        model |= feature_extraction.TFIDF(on='body')
        model |= preprocessing.StandardScaler()
        model |= linear_model.SoftmaxRegression()
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= preprocessing.StandardScaler()
        model |= linear_model.SoftmaxRegression()
    return model

def naive_bayes_ComplementNB(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):               
        model |= compose.Pipeline(
            ("tfidf", feature_extraction.TFIDF(on='body')),
            ("nb", naive_bayes.ComplementNB(alpha=1))
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= compose.Pipeline(
            #("tfidf", feature_extraction.TFIDF(on='body')),
            ("nb", naive_bayes.ComplementNB(alpha=1))
        )
    return model

def naive_bayes_GaussianNB(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):               
        model |= feature_extraction.TFIDF(on='body')
        model |= naive_bayes.GaussianNB()
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= naive_bayes.GaussianNB()
    return model

def naive_bayes_MultinomialNB(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):               
        model |= compose.Pipeline(
            ("tfidf", feature_extraction.TFIDF(on='body')),
            #("tokenize", feature_extraction.BagOfWords(lowercase=False)),
            ("nb", naive_bayes.MultinomialNB(alpha=1))
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        model |= compose.Pipeline(
            #("tfidf", feature_extraction.TFIDF(on='body')),
            #("tokenize", feature_extraction.BagOfWords(lowercase=False)),
            ("nb", naive_bayes.MultinomialNB(alpha=1))
        )
    return model

def neighbors_KNNClassifier(dataset):
    model=model_database(dataset)
    if isinstance(dataset, datasets.SMSSpam):               
        l1_dist = functools.partial(utils.math.minkowski_distance, p=1)
        model |= (feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            neighbors.KNNClassifier(
                engine=neighbors.SWINN(
                    dist_func=l1_dist,
                    seed=42
                )
            )
        )
    elif isinstance(dataset, datasets.TREC07):
        #model |= (
        #  feature_extraction.TFIDF(on='body') |
        #  naive_bayes.BernoulliNB(alpha=0)
        #)
        pass
    else:
        l1_dist = functools.partial(utils.math.minkowski_distance, p=1)
        model |= (#feature_extraction.TFIDF(on='body')|
            preprocessing.StandardScaler() |
            neighbors.KNNClassifier(
                engine=neighbors.SWINN(
                    dist_func=l1_dist,
                    seed=42
                )
            )
        )
    return model




