import click
import numpy as np
# import dill
from sklearn.linear_model import LogisticRegression
# from google.colab import files
from os import listdir
import pandas as pd
import boto3
import io
# from google.colab import files
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import cross_validate, StratifiedKFold
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# from sklearn.externals import joblib

def run_preprocess(s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_preprocess ...' )
    click.echo('s3_endpoint_url: %s' % s3_endpoint_url)        
    click.echo('s3_access_key: %s' % s3_access_key)
    click.echo('s3_secret_key: masked')
    click.echo('s3_bucket: %s' % s3_bucket)
    click.echo('pandas version: %s' % pd.__version__)

    s3 = boto3.client(service_name='s3',aws_access_key_id = s3_access_key, aws_secret_access_key = s3_secret_key, endpoint_url=s3_endpoint_url)

    response = s3.list_objects(Bucket=s3_bucket, Prefix="training_setA/", MaxKeys=max_keys)

    click.echo('File names found :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found : %s ' %file["Key"])
        
    df_list = []

    for file in response["Contents"]:
        obj = s3.get_object(Bucket=s3_bucket, Key=file["Key"])
        obj_df = pd.read_csv(obj["Body"], sep='|')
        df_list.append(obj_df)

    df = pd.concat(df_list)
    click.echo('Head records            :' )
    click.echo('%s' % df.head())
    

    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    click.echo('length of the all data  : %s' % len(df))
    click.echo('length of the train     : %s' % len(train))
    click.echo('length of the validate  : %s' % len(validate))
    click.echo('length of the test      : %s' % len(test))
    # Missing data
    missing = (train.isnull().sum() / train.shape[0]) * 100
    selected_cols = list(missing[missing < 90].index)
    click.echo('selected_cols           : %s' % selected_cols)
    filter_train = train[selected_cols]
    # BaseLine
    categorical = filter_train.dtypes == object
    categorical['Unit1'] = True
    categorical['Unit2'] = True
    categorical['Gender'] = True
    cat_vars = list(categorical[categorical].index)
    cont_vars = list(categorical[~categorical].index)
    click.echo('cont_vars               : %s' % cont_vars.pop(-1))
    features = cont_vars + cat_vars
    click.echo('features                : %s' % features)

    X_train = filter_train[features]
    y_train = filter_train['SepsisLabel']

    categorical = X_train.dtypes == object
    categorical['Unit1'] = True
    categorical['Unit2'] = True
    categorical['Gender'] = True

    cont_scale_pipeline = make_pipeline(SimpleImputer(strategy = "median"), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy = "constant", fill_value = 999), OneHotEncoder(handle_unknown="ignore"))
    preprocess_trans_scale = make_column_transformer((cont_scale_pipeline, ~categorical),(cat_pipeline, categorical))



def run_lr(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_lr ...' )
    logistic_pipe_scale = make_pipeline(preprocess_trans_scale, LogisticRegression(solver='lbfgs') )

    scores_logistic_pipe_scale = cross_validate(logistic_pipe_scale, X_train,  y_train, cv=StratifiedKFold(3, shuffle=True) ,scoring=["average_precision", "roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_logistic_pipe_scale)
    log_df['model'] = 'LogisticRegression'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'lrmodel.pkl') 
    lrmodel = joblib.load('lrmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/lrmodel.pkl"
    click.echo('uploading  lrmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="lrmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    


def run_rf(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_rf ...' )
    rf_pipe_scale = make_pipeline(preprocess_trans_scale, RandomForestClassifier())

    scores_rf_pipe_scale = cross_validate(rf_pipe_scale, X_train,y_train,cv=StratifiedKFold(3, shuffle=True),scoring=["average_precision","roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_rf_pipe_scale)
    log_df['model'] = 'RandomForest'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'rfmodel.pkl') 
    rfmodel = joblib.load('rfmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/rfmodel.pkl"
    click.echo('uploading  rfmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="rfmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    



def run_gb(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_gb ...' )
    gb_pipe_scale = make_pipeline(preprocess_trans_scale, GradientBoostingClassifier())

    scores_gb_pipe_scale  = cross_validate(gb_pipe_scale, X_train,y_train,cv=StratifiedKFold(3, shuffle=True),scoring=["average_precision","roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_gb_pipe_scale)
    log_df['model'] = 'GradientBoosting'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'gbmodel.pkl') 
    gbmodel = joblib.load('gbmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/gbmodel.pkl"
    click.echo('uploading  gbmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="gbmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    



def run_gmm(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_gmm ...' )
    gmm_pipe_scale = make_pipeline(preprocess_trans_scale, GaussianMixture(n_components=2))

    scores_gmm_pipe_scale  = cross_validate(gmm_pipe_scale, X_train,y_train,cv=StratifiedKFold(3, shuffle=True),scoring=["average_precision","roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_gmm_pipe_scale)
    log_df['model'] = 'GMM'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'gmmmodel.pkl') 
    gmmmodel = joblib.load('gmmmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/gmmmodel.pkl"
    click.echo('uploading  gmmmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="gmmmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    



def run_mlp(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_mlp ...' )
    mlp_pipe_scale = make_pipeline(preprocess_trans_scale, MLPClassifier())
    

    scores_mlp_pipe_scale  = cross_validate(mlp_pipe_scale, X_train,y_train,cv=StratifiedKFold(3, shuffle=True),scoring=["average_precision","roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_mlp_pipe_scale)
    log_df['model'] = 'MLP'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'mlpmodel.pkl') 
    mlpmodel = joblib.load('mlpmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/mlpmodel.pkl"
    click.echo('uploading  mlpmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="mlpmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    


def run_dt(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys):
    global X_train ,y_train, preprocess_trans_scale, s3
    click.echo('Begin run_dt ...' )
    dt_pipe_scale = make_pipeline(preprocess_trans_scale, DecisionTreeClassifier() )
    

    scores_dt_pipe_scale  = cross_validate(dt_pipe_scale, X_train,y_train,cv=StratifiedKFold(3, shuffle=True),scoring=["average_precision","roc_auc", "precision","recall"],return_estimator=True)

    log_df = pd.DataFrame(scores_dt_pipe_scale)
    log_df['model'] = 'DecisionTree'
    results = pd.concat([log_df])
    click.echo('results                 :')
    click.echo('%s' % results)

    results = pd.concat([log_df])
    results['mean'] = results.mean(axis=1)
    click.echo('results with mean       :')
    click.echo('%s' % results)

    click.echo('Selected model')
    click.echo('%s' %results.loc[results['mean'].idxmax()])
    bestmodel= results.loc[results['mean'].idxmax()]['estimator']
    joblib.dump(bestmodel, 'dtmodel.pkl') 
    dtmodel = joblib.load('dtmodel.pkl')
    click.echo('bestmodel.score   %s    :' %bestmodel.score(X_train,y_train))
    key = "uploaded/dtmodel.pkl"
    click.echo('uploading  dtmodel.pkl...')
    s3.upload_file(Bucket=s3_bucket, Key=key, Filename="dtmodel.pkl")
    response = s3.list_objects(Bucket=s3_bucket, Prefix="uploaded/", MaxKeys=max_keys)
    click.echo('Model names found       :')   
    toptenfiles = response["Contents"][:10]
    for file in toptenfiles:
        click.echo('File names found    : %s ' %file["Key"])    



@click.command()
@click.option('--s3_endpoint_url', envvar='S3_ENDPOINT_URL')
@click.option('--s3_access_key', envvar='S3_ACCESS_KEY')
@click.option('--s3_secret_key', envvar='S3_SECRET_KEY')
@click.option('--s3_bucket', envvar='S3_BUCKET')
@click.option('--max_keys', envvar='MAX_KEYS',type=int)
@click.option('--estimator', default='lr')
def run_pipeline(
        s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys,
        estimator):
    global X_train ,y_train, preprocess_trans_scale, s3

    run_preprocess(s3_endpoint_url, 
        s3_access_key,
        s3_secret_key, 
        s3_bucket,
        max_keys)

    if estimator == "lr" :
        run_lr(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    elif estimator == "rf" :
        run_rf(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    elif estimator == "gb" :
        run_gb(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    elif estimator == "gmm" :
        run_gmm(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    elif estimator == "mlp" :
        run_mlp(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    elif estimator == "dt" :
        run_dt(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)
    else :
        run_lr(s3_endpoint_url, 
            s3_access_key,
            s3_secret_key, 
            s3_bucket,
            max_keys)

if __name__ == "__main__":
    run_pipeline()

