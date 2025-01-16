from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
# helper function to calculate per class f1 scores


def per_class_f1_score(actual_classes, prediction_classes):
    # Compute F1 scores for each class directly
    f1_scores = f1_score(actual_classes, prediction_classes, average=None)
    # Create a list of dictionaries for output
    per_class_f1_scores = [{"class": i, "f1_score": score}
                           for i, score in enumerate(f1_scores)]

    return per_class_f1_scores


def update_model_comparison(probabilities, true_labels, algorithm, method, filename, comparison_df=None):
    """
    Updates the model comparison DataFrame with metrics for a given model.

    Args:
        probabilities (ndarray): Probabilities or predicted values for the dataset.
        true_labels (Series or ndarray): True labels for the dataset.
        algorithm (str): Name of the algorithm (e.g., 'Logistic Regression').
        method (str): Method used (e.g., 'Default Params', 'Grid Search').
        filename (str): Name of the file to save the model.
        comparison_df (DataFrame or None): Existing comparison DataFrame. If None, a new one is created.

    Returns:
        DataFrame: Updated comparison DataFrame with metrics for the given model.
    """

    # Get predicted classes (argmax for probabilities)
    predicted_classes = probabilities.argmax(axis=1)

    # Compute metrics
    accuracy = accuracy_score(true_labels, predicted_classes)
    weighted_f1 = f1_score(true_labels, predicted_classes, average='weighted')
    roc_score = roc_auc_score(true_labels, probabilities, multi_class="ovr")
    # Compute per-class F1 scores
    per_class_f1_scores = f1_score(
        true_labels, predicted_classes, average=None)
    per_class_f1_dict = {f"Class_{i}": score for i,
                         score in enumerate(per_class_f1_scores)}

    # Create a new row with metrics
    new_row = {
        "Algorithm": algorithm,
        "Method": method,
        "File Name": filename,
        "Accuracy": accuracy,
        "Weighted F1 Score": weighted_f1,
        "ROC AUC Score": roc_score,
        **per_class_f1_dict,  # Unpack per-class F1 scores
    }

    # Initialize or update the DataFrame
    if comparison_df is None:
        return pd.DataFrame([new_row])

    # Append the new row
    comparison_df = pd.concat(
        [comparison_df, pd.DataFrame([new_row])], ignore_index=True)

    return comparison_df
