import logging
import azure.functions as func
import sys
import json

sys.path.insert(0,'predict') #insert utils into root to allow joblib to load (???)
from .utils import *
import shap
import joblib


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def single_force_plot(i, html=True):
#     if html:
#         fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
#             feature_names=feat_used, show=False, link='logit')
#         shap.save_html('./result/shap_force_plot_' + str(i) + '.htm', fig)
#     else:
#         fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
#             feature_names=feat_used, show=False, matplotlib=True, link='logit')
#         # fig = plt.gcf()
#         # fig.savefig('./result/shap_force_plot_' + str(i) + '.svg')
#         # fig.close()
#     return fig

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    logging.info(sys.version)
    logging.info(np.__version__)
    logging.info(pd.__version__)

    data_pipeline = joblib.load('predict/XGB_pipeline.joblib')

    model = data_pipeline.model
    imputer = data_pipeline.imputer
    scaler = data_pipeline.normalizer
    feat_used = data_pipeline.feat_used

    logging.info(model)
    explainer = shap.TreeExplainer(model)

    # this is made up patient data; change to make new predictions
    example_patient = {# Most important variables
                    'percent_transfused': 0.9, # historical transfusion % for surgery patient is having
                    'PRHCT' : 30,              # Hematocrit in g/dL
                    'ASA': 3,                  # ASA physical status
                    'WEIGHT' : 190,            # Weight in lbs
                    # Preop laboratory values
                    'PRPLATE' : 100,
                    'PRINR' : 2,
                    'PRPTT' : 33,
                    'PRCREAT' : 1.0,
                    'PRSODM' : 140,
                    'PRALBUM' : 4.0,
                    'PRBILI' : 1.0, 
                    # Patient demographics and comorbidities
                    'Age': 70,
                    'HEIGHT' : np.nan,
                    'ELECTSURG' : 1,
                    'SEX' : 1, # female = 1
                    'HYPERMED' : 1,
                    'DIALYSIS' : 0,
                    'HXCHF' : 0, 
                    'HXCOPD' : 1, 
                    'DIABETES' : 1, 
                    'SMOKE' : 1, 
                    }
    example_data = pd.DataFrame(example_patient, index=[0])
    logging.info(example_data)

    # transform data
    data_to_explain = example_data[feat_used]
    data_to_explain = pd.DataFrame(imputer.transform(data_to_explain), columns=feat_used)
    data_scaled = pd.DataFrame(scaler.transform(data_to_explain), columns=feat_used)

    # # print predicted probability of transfusion for the above example patient
    # # logging.info(predict_example(model, imputer, scaler, example_data.loc[0, feat_used]))
    vec = np.array(example_data.loc[0, feat_used]).reshape(1, -1)
    if imputer != None:
        vec = imputer.transform(vec)
    vec = scaler.transform(vec)
    y_prob = model.predict_proba(vec)[:, 1]
    logging.info(y_prob[0])

    # calculate shap values for above example
    shap_values = explainer.shap_values(data_scaled)

    shap.initjs()


    result = shap.force_plot(explainer.expected_value, 
                shap_values[0, :], data_to_explain.iloc[0, :],
                feature_names=feat_used, link='logit')

    i = 0
    # html = True
    # if html:
    #     fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
    #         feature_names=feat_used, show=False, link='logit')
    #     shap.save_html('shap_force_plot.htm', fig)
        
    # else:
    #     fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
    #         feature_names=feat_used, show=False, matplotlib=True, link='logit')
    
    
    output = {"img": "", "prob":str(y_prob[0])}
    return func.HttpResponse(json.dumps(output))
