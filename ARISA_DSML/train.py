"""Functions to train model."""
from pathlib import Path

from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
from mlflow.client import MlflowClient
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    categorical,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash
import nannyml as nml


# comment to trigger workflow ver6

def run_hyperopt(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int], test_size:float=0.25, n_trials:int=20, overwrite:bool=False)->str|Path:  # noqa: PLR0913
    """Run optuna hyperparameter tuning."""
    logger.info("Running hyperparameter tuning")
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        def objective(trial:optuna.trial.Trial)->float:
            with mlflow.start_run(nested=True):
                params = {
                    "depth": trial.suggest_int("depth", 2, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1),
                    "iterations": trial.suggest_int("iterations", 50, 400),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
                    "ignored_features": [],
                }
                model = CatBoostClassifier(**params, verbose=0)
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    cat_features=categorical_indices,
                    early_stopping_rounds=50,
                )
                mlflow.log_params(params)
                preds = model.predict(X_val_opt)
                probs = model.predict_proba(X_val_opt)

                f1 = f1_score(y_val_opt, preds, average="macro")
                logloss = log_loss(y_val_opt, probs)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("logloss", logloss)

            return model.get_best_score()["validation"]["MultiClass"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study.best_params, best_params_path)

        params = study.best_params
    else:
        params = joblib.load(best_params_path)
    logger.info("Best Parameters: " + str(params))
    return best_params_path


def train_cv(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int], params:dict, eval_metric:str="TotalF1", n:int=5)->str|Path:  # noqa: PLR0913
    """Do cross-validated training."""
    logger.info("Running cross-validation")
    params["eval_metric"] = eval_metric
    params["loss_function"] = "MultiClass"
    params["custom_metric"] = ["F1"]
    params["ignored_features"] = []

    data = Pool(X_train, y_train, cat_features=categorical_indices)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
        plot=True,

    )

    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)

    return cv_output_path


def train(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int],  # noqa: PLR0913
          params:dict|None, artifact_name:str="catboost_model_wine", cv_results=None,
          )->tuple[str|Path]:
    """Train model on full dataset without cross-validation."""
    logger.info("Running training")
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}
    with mlflow.start_run():
        params["ignored_features"] = []

        model = CatBoostClassifier(
            **params,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            cat_features=categorical_indices,
            use_best_model=False,
            plot=True,
        )
        params["feature_columns"] = X_train.columns
        mlflow.log_params(params)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)
        if "test-TotalF1-mean" in cv_results.columns:
            cv_metric_mean = cv_results["test-TotalF1-mean"].mean()
        elif "test-F1-mean" in cv_results.columns:
            cv_metric_mean = cv_results["test-F1-mean"].mean()
        else:
            logger.warning("F1 metric not found in cv_results, logging NaN.")
            cv_metric_mean = float("nan")
        mlflow.log_metric("f1_cv_mean", cv_metric_mean)

        # Log the model
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            input_example=X_train,
            registered_model_name=MODEL_NAME,
        )
        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions(MODEL_NAME)[0]
        client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )
        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(params, model_params_path)
        fig1 = plot_error_scatter(
            df_plot=cv_results,
            name="Mean F1 Score",
            title="Cross-Validation (N=5) Mean F1 score with Error Bands",
            xtitle="Training Steps",
            ytitle="Performance Score",
        )
        mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")
        logger.info("test-F1-mean vs iterations plot saved")

        fig2 = plot_error_scatter(
            cv_results,
            x="iterations",
            y="test-MultiClass-mean",
            err="test-MultiClass-std",
            name="Mean MultiClass Logloss",
            title="Cross-Validation (N=5) Mean MultiClass Logloss with Error Bands",
            xtitle="Training Steps",
            ytitle="MultiClass Logloss",
        )
        mlflow.log_figure(fig2, "test-multiclass-logloss-mean_vs_iterations.png")
        logger.info("test-MultiClass-mean vs iterations plot saved")

        """----------NannyML----------"""
        # Model monitoring initialization
        proba_col_names = [f"proba_{cls}" for cls in model.classes_]
        reference_df = X_train.copy()
        reference_df["prediction"] = model.predict(X_train)
        proba_df = pd.DataFrame(model.predict_proba(X_train), columns=proba_col_names)
        reference_df = pd.concat([reference_df, proba_df], axis=1)
        reference_df[target] = y_train
        chunk_size = 50

        # univariate drift for features
        udc = nml.UnivariateDriftCalculator(
            column_names=X_train.columns,
            chunk_size=chunk_size,
        )
        udc.fit(reference_df.drop(columns=["prediction", target]))

        # Confidence-based Performance Estimation for target
        y_pred_proba_dict = {cls: f"proba_{cls}" for cls in model.classes_}
        estimator = nml.CBPE(
            problem_type="classification_multiclass",
            y_pred_proba=y_pred_proba_dict,
            y_pred="prediction",
            y_true=target,
            metrics=["roc_auc"],
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")

        mlflow.log_artifact(MODELS_DIR / "udc.pkl")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl")

        logger.info(f"quality_label dtype in reference: {reference_df['quality_label'].dtype}")
        logger.info(f"Reference value counts: {reference_df['quality_label'].value_counts(dropna=False)}")

        n_chunks = int(len(reference_df) / chunk_size)

        for i in range(n_chunks):
            chunk = reference_df.iloc[i*chunk_size: (i+1)*chunk_size]
            value_counts = chunk["quality_label"].value_counts(dropna=False)
            logger.info(f"Chunk {i+1}/{n_chunks} value counts:\n{value_counts}")

    return (model_path, model_params_path)


def plot_error_scatter(  # noqa: PLR0913
        df_plot:pd.DataFrame,
        x:str="iterations",
        y:str="test-TotalF1-mean",
        err:str="test-TotalF1-std",
        name:str="",
        title:str="",
        xtitle:str="",
        ytitle:str="",
        yaxis_range:list[float]|None=None,
    )->None:
    """Plot plotly scatter plots with error areas."""
    logger.info("Plotting error scatter")
    # Create figure
    fig = go.Figure()

    if not len(name):
        name = y

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[y], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y]+df_plot[err],
                         df_plot[y]-df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color":"rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    # fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig


def get_or_create_experiment(experiment_name:str):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters
    ----------
    - experiment_name (str): Name of the MLflow experiment.

    Returns
    -------
    - str: ID of the existing or newly created MLflow experiment.

    """
    logger.info(f"Retrieving or creating experiment: {experiment_name}")
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


# def champion_callback(study, frozen_trial):
#     """
#     Logging callback that will report when a new trial iteration improves upon existing
#     best trial values.

#     Note: This callback is not intended for use in distributed computing systems such as Spark
#     or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
#     workers or agents.
#     The race conditions with file system state management for distributed trials will render
#     inconsistent values with this callback.
#     """

#     winner = study.user_attrs.get("winner", None)

#     if study.best_value and winner != study.best_value:
#         study.set_user_attr("winner", study.best_value)
#         if winner:
#             improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
#             print(
#                 f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
#                 f"{improvement_percent: .4f}% improvement"
#             )
#         else:
#             print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


if __name__=="__main__":
    # for running in workflow in actions again again
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    y_train = df_train.pop(target)
    X_train = df_train

    categorical_indices = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]
    experiment_id = get_or_create_experiment("wine_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train, categorical_indices)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, categorical_indices, params)
    cv_results = pd.read_csv(cv_output_path)

    experiment_id = get_or_create_experiment("wine_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(X_train, y_train, categorical_indices, params, cv_results=cv_results)

    cv_results = pd.read_csv(cv_output_path)
