"""Run prediction on test data."""
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import shap
import joblib
import os
from ARISA_DSML.config import FIGURES_DIR, MODELS_DIR, target, PROCESSED_DATA_DIR, categorical, MODEL_NAME
from ARISA_DSML.resolve import get_model_by_alias
import mlflow
from mlflow.client import MlflowClient
import json
import nannyml as nml


def plot_shap(model:CatBoostClassifier, df_plot:pd.DataFrame)->None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")


def predict(model:CatBoostClassifier, df_pred:pd.DataFrame, params:dict, probs=False)->str|Path:
    """Do predictions on test data."""
    
    feature_columns = params.pop("feature_columns")
    
    preds = model.predict(df_pred[feature_columns])
    proba_cols = [f"proba_{cls}" for cls in model.classes_]

    if probs:
        proba_df = pd.DataFrame(model.predict_proba(df_pred[feature_columns]), columns=proba_cols)
        df_pred = pd.concat([df_pred, proba_df], axis=1)

    plot_shap(model, df_pred[feature_columns])
    df_pred[target] = preds
    preds_path = MODELS_DIR / "preds.csv"
    if not probs:
        df_pred[[target]].to_csv(preds_path, index=True)
    else:
        df_pred[[target] + proba_cols].to_csv(preds_path, index=True)

    return preds_path


if __name__=="__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    if model_info is None:
        logger.info("No champion model, predicting using newest model")
        model_info = client.get_latest_versions(MODEL_NAME)[0]

    # extract params/metrics data for run `test_run_id` in a single dict
    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(run.data.tags['mlflow.log-model.history'])
    log_model_meta[0]['signature']


    _, artifact_folder = os.path.split(model_info.source)
    logger.info(artifact_folder)
    model_uri = "runs:/{}/{}".format(model_info.run_id, artifact_folder)
    logger.info(model_uri)
    loaded_model = mlflow.catboost.load_model(model_uri)

    local_path = client.download_artifacts(model_info.run_id, "udc.pkl", "models")
    local_path = client.download_artifacts(model_info.run_id, "estimator.pkl", "models")

    store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
    udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
    estimator = store.load(filename="estimator.pkl", as_type=nml.CBPE)
    
    logger.info(f"Loaded UDC chunk_size: {udc.chunker.chunk_size}")

    params = run_data_dict["params"]
    params["feature_columns"] = [inp["name"] for inp in json.loads(log_model_meta[0]['signature']['inputs'])]
    preds_path = predict(loaded_model, df_test, params, probs=True)
    
    df_preds = pd.read_csv(preds_path)

    analysis_df = df_test.copy()
    analysis_df["prediction"] = df_preds[target]
    proba_cols = [col for col in df_preds.columns if col.startswith("proba_")]
    for col in proba_cols:
        analysis_df[col] = df_preds[col]

    logger.info(f"Analysis value counts: {analysis_df['quality_label'].value_counts(dropna=False)}")
    #n_chunks = int(np.ceil(len(analysis_df) / udc.chunker.chunk_size))
    logger.info(f"quality_label dtype in analysis: {analysis_df['quality_label'].dtype}")
    n_chunks = int(len(analysis_df) / udc.chunker.chunk_size)

    for i in range(n_chunks):
        chunk = analysis_df.iloc[i*udc.chunker.chunk_size : (i+1)*udc.chunker.chunk_size]
        value_counts = chunk["quality_label"].value_counts(dropna=False)
        logger.info(f"Chunk {i+1}/{n_chunks} value counts:\n{value_counts}")

    from ARISA_DSML.train import get_or_create_experiment
    from ARISA_DSML.helpers import get_git_commit_hash

    git_hash = get_git_commit_hash()
    logger.info(f"Git hash: {git_hash}")

    mlflow.set_experiment("wine_predictions")
    with mlflow.start_run(tags={"git_sha": get_git_commit_hash()}):
        estimated_performance = estimator.estimate(analysis_df)
        fig1 = estimated_performance.plot()
        mlflow.log_figure(fig1, "estimated_performance.png")

        for col in analysis_df.drop(columns=["prediction"] + proba_cols):
            unique_vals = analysis_df[col].nunique(dropna=True)
            logger.info(f"Column {col} unique values: {unique_vals}")

        drop_cols = ["prediction"] + proba_cols + [target]
        univariate_drift = udc.calculate(analysis_df.drop(columns=drop_cols, axis=1))
        plot_col_names = analysis_df.drop(columns=drop_cols, axis=1).columns
        
        logger.info(f"Univariate drift columns: {plot_col_names}")

        for p in plot_col_names:
            try:
                fig2 = univariate_drift.filter(column_names=[p]).plot()
                mlflow.log_figure(fig2, f"univariate_drift_{p}.png")
                fig3 = univariate_drift.filter(period="analysis", column_names=[p]).plot(kind='distribution')
                mlflow.log_figure(fig3, f"univariate_drift_dist_{p}.png")
            except Exception as e:
                logger.exception(f"Failed to plot univariate drift for column {p}: {e}")
        mlflow.log_params({"git_hash": git_hash})