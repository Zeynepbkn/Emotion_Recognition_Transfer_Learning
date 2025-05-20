"""
Optimized Emotion Recognition Model Evaluation
This script provides comprehensive evaluation of the trained emotion recognition model
with detailed metrics, visualizations, and performance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def load_model(model_path='optimized_emotion_classifier.pkl'):
    """
    Loads a trained model from disk
    """
    try:
        learn = load_learner(model_path)
        print(f"Model successfully loaded: {model_path}")
        return learn
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_accuracy(learn, test_path=None):
    """
    Evaluates model accuracy on test data
    """
    print("=" * 50)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 50)
    
    # Use default test path if none provided
    if test_path is None:
        test_path = Path("EMOTION RECOGNITION DATASET")
    
    # Create test dataloader with labels
    dls = learn.dls.test_dl(get_image_files(test_path), with_labels=True, num_workers=0)
    
    # Get predictions and targets
    preds, targets = learn.get_preds(dl=dls)
    pred_classes = preds.argmax(dim=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(targets, pred_classes)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = learn.dls.vocab
    report = classification_report(targets, pred_classes, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save report to file
    with open('evaluation_report.txt', 'w') as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    return preds, targets, class_names

def visualize_confusion_matrix(targets, pred_classes, class_names):
    """
    Creates and saves a detailed confusion matrix
    """
    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, pred_classes)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('evaluation_confusion_matrix.png')
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('evaluation_normalized_confusion_matrix.png')
    plt.close()
    
    return cm, cm_norm

def visualize_roc_curves(preds, targets, class_names):
    """
    Plots ROC curves for multi-class classification
    """
    print("=" * 50)
    print("ROC CURVES")
    print("=" * 50)
    
    # Convert targets to one-hot encoding
    n_classes = len(class_names)
    targets_one_hot = label_binarize(targets, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('evaluation_roc_curves.png')
    plt.close()
    
    return roc_auc

def visualize_precision_recall(preds, targets, class_names):
    """
    Plots Precision-Recall curves for multi-class classification
    """
    print("=" * 50)
    print("PRECISION-RECALL CURVES")
    print("=" * 50)
    
    # Convert targets to one-hot encoding
    n_classes = len(class_names)
    targets_one_hot = label_binarize(targets, classes=range(n_classes))
    
    # Compute Precision-Recall curve and average precision for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(targets_one_hot[:, i], preds[:, i])
        avg_precision[i] = average_precision_score(targets_one_hot[:, i], preds[:, i])
        
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.savefig('evaluation_precision_recall_curves.png')
    plt.close()
    
    return avg_precision

def analyze_prediction_confidence(preds, targets, class_names):
    """
    Analyzes prediction confidence distributions for correct and incorrect predictions
    """
    print("=" * 50)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 50)
    
    # Get highest probability for each prediction
    pred_classes = preds.argmax(dim=1)
    pred_probs = preds.max(dim=1)[0]
    
    # Separate probabilities for correct and incorrect predictions
    correct_mask = pred_classes == targets
    correct_probs = pred_probs[correct_mask]
    incorrect_probs = pred_probs[~correct_mask]
    
    # Plot confidence distributions
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    if len(correct_probs) > 0:
        plt.hist(correct_probs.numpy(), alpha=0.5, bins=20, 
                 label=f'Correct predictions (n={len(correct_probs)})')
    
    if len(incorrect_probs) > 0:
        plt.hist(incorrect_probs.numpy(), alpha=0.5, bins=20,
                 label=f'Incorrect predictions (n={len(incorrect_probs)})')
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Confidence')
    plt.legend()
    plt.savefig('evaluation_confidence_distribution.png')
    plt.close()
    
    # Calculate statistics
    if len(correct_probs) > 0:
        avg_correct_conf = correct_probs.mean().item()
        print(f"Average confidence for correct predictions: {avg_correct_conf:.4f}")
    else:
        avg_correct_conf = 0
        print("No correct predictions found")
    
    if len(incorrect_probs) > 0:
        avg_incorrect_conf = incorrect_probs.mean().item()
        print(f"Average confidence for incorrect predictions: {avg_incorrect_conf:.4f}")
    else:
        avg_incorrect_conf = 0
        print("No incorrect predictions found")
    
    return avg_correct_conf, avg_incorrect_conf

def visualize_sample_predictions(learn, test_path=None, num_samples=10):
    """
    Visualizes sample predictions with probabilities
    """
    print("=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    
    if test_path is None:
        test_path = Path("EMOTION RECOGNITION DATASET")
    
    # Get random test images
    test_files = get_image_files(test_path)
    if len(test_files) > num_samples:
        test_files = np.random.choice(test_files, num_samples, replace=False)
    
    # Create figure for visualization
    rows = min(3, num_samples)
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Make predictions and plot
    for i, img_path in enumerate(test_files):
        if i >= len(axes):
            break
            
        # Load image and predict
        img = PILImage.create(img_path)
        pred_class, pred_idx, probs = learn.predict(img)
        
        # Get true label from parent directory
        true_label = img_path.parent.name
        
        # Plot image
        axes[i].imshow(img)
        
        # Set title color based on correctness
        if true_label == pred_class:
            title_color = 'green'
        else:
            title_color = 'red'
        
        # Set title with prediction info
        axes[i].set_title(f"Pred: {pred_class} ({probs[pred_idx]:.2f})\nTrue: {true_label}", 
                        color=title_color)
        axes[i].axis('off')
    
    # Remove unused axes
    for i in range(len(test_files), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('evaluation_sample_predictions.png')
    plt.close()

def evaluate_class_wise_metrics(targets, pred_classes, class_names):
    """
    Calculates and visualizes class-wise performance metrics
    """
    print("=" * 50)
    print("CLASS-WISE PERFORMANCE METRICS")
    print("=" * 50)
    
    # Calculate per-class metrics from confusion matrix
    cm = confusion_matrix(targets, pred_classes)
    
    # Per-class accuracy (diagonal divided by row sum)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    # Per-class precision (diagonal divided by column sum)
    class_precision = np.diag(cm) / np.sum(cm, axis=0)
    
    # Per-class recall (same as per-class accuracy)
    class_recall = class_accuracy
    
    # Per-class F1 score
    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
    
    # Create dataframe for metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': class_accuracy,
        'Precision': class_precision,
        'Recall': class_recall,
        'F1 Score': class_f1
    })
    
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv('class_wise_metrics.csv', index=False)
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    # Create bar positions
    x = np.arange(len(class_names))
    width = 0.2
    
    # Plot bars
    plt.bar(x - width*1.5, class_accuracy, width, label='Accuracy')
    plt.bar(x - width/2, class_precision, width, label='Precision')
    plt.bar(x + width/2, class_recall, width, label='Recall')
    plt.bar(x + width*1.5, class_f1, width, label='F1')
    
    # Add labels and legend
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Class-wise Performance Metrics')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation_class_wise_metrics.png')
    plt.close()
    
    return metrics_df

def main():
    """
    Main evaluation function
    """
    print("OPTIMIZED EMOTION RECOGNITION MODEL EVALUATION")
    print("=" * 50)
    
    # Load trained model
    model_path = 'optimized_emotion_classifier.pkl'
    learn = load_model(model_path)
    
    if learn is None:
        print("Failed to load model. Trying backup model...")
        model_path = 'emotion_classifier.pkl'
        learn = load_model(model_path)
        
        if learn is None:
            print("Could not load any model. Evaluation aborted.")
            return
    
    # Evaluate model accuracy
    preds, targets, class_names = evaluate_accuracy(learn)
    pred_classes = preds.argmax(dim=1)
    
    # Visualize confusion matrix
    cm, cm_norm = visualize_confusion_matrix(targets, pred_classes, class_names)
    
    # Calculate ROC curves
    roc_auc = visualize_roc_curves(preds, targets, class_names)
    
    # Calculate precision-recall curves
    avg_precision = visualize_precision_recall(preds, targets, class_names)
    
    # Analyze prediction confidence
    avg_correct_conf, avg_incorrect_conf = analyze_prediction_confidence(preds, targets, class_names)
    
    # Visualize sample predictions
    visualize_sample_predictions(learn, num_samples=9)
    
    # Calculate class-wise metrics
    metrics_df = evaluate_class_wise_metrics(targets, pred_classes, class_names)
    
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = accuracy_score(targets, pred_classes)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Average metrics across classes
    print(f"Average ROC AUC: {np.mean(list(roc_auc.values())):.4f}")
    print(f"Average Precision: {np.mean(list(avg_precision.values())):.4f}")
    
    # Confidence gap
    conf_gap = avg_correct_conf - avg_incorrect_conf
    print(f"Confidence gap (correct-incorrect): {conf_gap:.4f}")
    
    print("\nClass-wise F1 scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics_df['F1 Score'][i]:.4f}")
    
    print("\nEvaluation completed. Results saved to files.")

if __name__ == "__main__":
    main() 