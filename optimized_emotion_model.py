"""
Optimized Emotion Recognition Model Training
This script implements a comprehensive deep learning training pipeline with
optimized parameters for emotion recognition tasks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from fastai.metrics import accuracy, error_rate

accuracy_metric = accuracy
error_rate_metric = error_rate

# A.1. Download the data
# In this case, we're using the existing EMOTION RECOGNITION DATASET
def inspect_dataset(data_path):
    """
    A.1.1. Inspect the data layout
    Analyzes the dataset structure and distribution
    """
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)
    
    # Check available classes
    classes = os.listdir(data_path)
    print(f"Classes in the dataset: {classes}")
    
    # Count images per class
    class_counts = {}
    total_images = 0
    for emotion in classes:
        files = os.listdir(data_path/emotion)
        class_counts[emotion] = len(files)
        total_images += len(files)
        print(f"{emotion} class: {len(files)} images")
    
    print(f"Total images: {total_images}")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution.png')
    plt.close()
    
    return classes, class_counts

# A.2. Create the DataBlock and dataloaders
def create_datablock(data_path, img_size=224, batch_size=64, valid_pct=0.2):
    """
    Creates an optimized DataBlock for training
    A.1.2 Decision on how to create datablock based on dataset structure
    A.2.1 It defined the blocks
    A.2.2 It defined the means of getting data into DataBlock
    A.2.3 It defined how to get the attributes
    A.2.4 It defined data transformations with presizing
    """
    print("=" * 50)
    print("CREATING DATABLOCK")
    print("=" * 50)
    
    # A.1.2 Decision: We'll use parent folder name as the label for classification
    print("A.1.2: Using folder structure for class labels, with parent_label")
    
    # Define the DataBlock with all required components
    emotion_data = DataBlock(
        # A.2.1 Define blocks (input and target types)
        blocks=(ImageBlock, CategoryBlock),
        
        # A.2.2 Define how to get data
        get_items=get_image_files,
        
        # Data splitting strategy - random with fixed seed for reproducibility
        splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
        
        # A.2.3 Define how to get labels (from parent folder name)
        get_y=parent_label,
        
        # A.2.4 Define transformations with presizing strategy
        # First resize (item by item)
        item_tfms=[Resize(img_size, method='squish')],
        
        # Then apply augmentations (batch by batch)
        batch_tfms=[
            # Augmentations (applied to batch)
            *aug_transforms(size=img_size-32, min_scale=0.75, 
                           flip_vert=False, max_rotate=10.0, max_zoom=1.1),
            # Normalize using ImageNet stats
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    
    # Create dataloaders
    print(f"Creating dataloaders with batch size: {batch_size}")
    dls = emotion_data.dataloaders(data_path, bs=batch_size, num_workers=0)
    
    return emotion_data, dls

# A.3. Inspect the DataBlock via dataloader
def inspect_datablock(emotion_data, dls, data_path):
    """
    Inspects the created DataBlock and visualizes samples
    A.3.1 Show batch
    A.3.2 Check the labels
    A.3.3 Summarize the DataBlock
    """
    print("=" * 50)
    print("INSPECTING DATABLOCK")
    print("=" * 50)
    
    # A.3.1 Show batch
    print("Displaying sample batch...")
    dls.show_batch(max_n=9, figsize=(12, 10))
    plt.savefig('sample_batch.png')
    plt.close()
    
    # A.3.2 Check labels
    print(f"Classes (labels): {dls.vocab}")
    
    # A.3.3 DataBlock summary
    print("DataBlock summary:")
    emotion_data.summary(data_path)
    
    return dls.vocab

# A.4. Train a simple model (benchmark)
def train_benchmark_model(dls, model_name='resnet18'):
    """
    A.4.1 Create a benchmark model for comparison
    """
    print("=" * 50)
    print(f"TRAINING BENCHMARK MODEL: {model_name}")
    print("=" * 50)
    
    # Create learner with simple architecture
    learn = vision_learner(dls, 
                      eval(model_name), 
                      metrics=[error_rate_metric, accuracy_metric])  
    
    # Quick training for benchmark
    learn.fine_tune(3, base_lr=1e-2)
    
    # Save benchmark results
    learn.save('benchmark_model')
    
    # A.4.2 & A.4.3 Interpret model and create confusion matrix
    interpret_model(learn, "benchmark")
    
    return learn

# Helper function to interpret model performance
def interpret_model(learn, name_prefix=""):
    """
    A.4.2 Interpret the model
    A.4.3 Confusion matrix
    Creates visualizations to understand model performance
    """
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    interp.plot_confusion_matrix(figsize=(10, 8))
    plt.savefig(f'{name_prefix}_confusion_matrix.png')
    plt.close()
    
    # Plot top losses
    interp.plot_top_losses(9, figsize=(12, 10))
    plt.savefig(f'{name_prefix}_top_losses.png')
    plt.close()
    
    # Compute classification report
    probs, targets = learn.get_preds()
    preds = probs.argmax(dim=1)
    report = classification_report(targets, preds, target_names=learn.dls.vocab)
    print(f"Classification Report:\n{report}")
    
    # Save report to file
    with open(f'{name_prefix}_report.txt', 'w') as f:
        f.write(report)
    
    return interp

# B.1 and B.2 Learning Rate Finder
def find_learning_rate(learn):
    """
    B.1 & B.2 Implements learning rate finder
    Helps find optimal learning rate for training
    """
    print("=" * 50)
    print("FINDING OPTIMAL LEARNING RATE")
    print("=" * 50)
    
    # Run learning rate finder
    lr_suggestions = learn.lr_find(suggest_funcs=(valley, slide))
    
    # Plot and save results
    plt.savefig('learning_rate_finder.png')
    plt.close()
    
    print(f"Suggested learning rates: {lr_suggestions}")
    
    return lr_suggestions

# Custom manual learning rate finder implementation
def manual_lr_finder(learn, start_lr=1e-7, factor=2, max_iterations=100, max_lr=10):
    """
    B.2 Custom implementation of learning rate finder
    B.2.1 Start with a very very low lr
    B.2.2 Train one batch with lr, record loss
    B.2.3 Increase lr to 2*lr
    B.2.4 Train one batch with 2*lr, record the new loss
    B.2.5 If the new loss is smaller than old loss, continue increasing
    
    Args:
        learn: fastai Learner object
        start_lr: initial learning rate (very low)
        factor: multiplier for increasing learning rate
        max_iterations: maximum number of iterations
        max_lr: maximum learning rate to try
    
    Returns:
        optimal_lr: the learning rate that gave the best loss
        lrs: list of all learning rates tried
        losses: list of corresponding losses
    """
    print("=" * 50)
    print("RUNNING MANUAL LEARNING RATE FINDER")
    print("=" * 50)
    
    # Save the current learning rate and model weights
    original_lr = learn.opt.hypers[0]['lr']
    learn.save('temp_weights')
    
    # Initialize variables
    lr = start_lr
    lrs = []
    losses = []
    best_loss = float('inf')
    optimal_lr = start_lr
    
    # Get a single batch from training data
    xb, yb = next(iter(learn.dls.train))
    
    print(f"Starting with learning rate: {lr}")
    
    # Implement learning rate finder algorithm
    for i in range(max_iterations):
        # B.2.1 & B.2.3: Set the current learning rate
        learn.opt.set_hyper('lr', lr)
        
        # B.2.2 & B.2.4: Train on one batch and record loss
        learn.opt.zero_grad()
        loss = learn.loss_func(learn.model(xb), yb)
        loss.backward()
        learn.opt.step()
        
        # Record results
        lrs.append(lr)
        current_loss = loss.item()
        losses.append(current_loss)
        
        print(f"Iteration {i+1}: lr={lr:.8f}, loss={current_loss:.4f}")
        
        # Update best learning rate if this loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            optimal_lr = lr
        
        # Check if loss is NaN or too large (diverging)
        if np.isnan(current_loss) or current_loss > 4 * best_loss:
            print(f"Stopping early: loss is {'NaN' if np.isnan(current_loss) else 'diverging'}")
            break
            
        # B.2.3 & B.2.5: Increase lr by factor for next iteration
        lr *= factor
        
        # Stop if lr exceeds max_lr
        if lr > max_lr:
            print(f"Reached maximum learning rate: {max_lr}")
            break
    
    # Plot learning rate vs loss
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Manual Learning Rate Finder')
    plt.axvline(x=optimal_lr, color='r', linestyle='--')
    plt.savefig('manual_lr_finder.png')
    plt.close()
    
    # Restore original settings
    learn.opt.set_hyper('lr', original_lr)
    learn.load('temp_weights')
    
    print(f"Optimal learning rate found: {optimal_lr}")
    
    return optimal_lr, lrs, losses

# B.3 Implementation of Transfer Learning strategies
def compare_transfer_learning_strategies(dls, model_arch='resnet34', lr=1e-3, epochs=1):
    """
    B.3 Compares different transfer learning strategies:
    
    B.3.1 Replace the trained final linear layer F with a new one F' 
         (where size changes from mxn to mxk, k is the number of classes in new task)
    B.3.2 Only train F' while keeping previous weights frozen
    B.3.3 Continue training all weights (F' and previous) by unfreezing
    
    Args:
        dls: DataLoaders object
        model_arch: Model architecture to use
        lr: Learning rate to use
        epochs: Number of epochs for each strategy
        
    Returns:
        best_strategy: The learner with the best strategy applied
        results: Dictionary with results of all strategies
    """
    print("=" * 50)
    print("COMPARING TRANSFER LEARNING STRATEGIES")
    print("=" * 50)
    
    results = {}
    
    # Strategy 1: Train only the head (final layer) with frozen body
    print("\nStrategy 1: Train only the head (final layer) with frozen body")
    learn1 = vision_learner(dls, eval(model_arch), metrics=[error_rate_metric, accuracy_metric], pretrained=True) # B.3.3 vision_learner fonksiyonu çağrıldığında, önceden eğitilmiş bir model yüklenir ve son katmanı (head) otomatik olarak yeni görevin sınıf sayısına uygun şekilde değiştirilir. Bu, tam olarak eski son doğrusal katman F'yi (boyutu mxn) yeni bir F' (boyutu mxk) ile değiştirme işlemidir.
    
    # The model comes with pre-trained weights, and the head is already replaced with a new one
    # matching the number of classes in our task (this happens automatically in vision_learner)
    print("Model summary before training:")
    print(learn1.model)
    
    # Train only the head (Freezing)
    learn1.fit_one_cycle(epochs, lr) #B.3. Only train F' while using the previous weights unchanged (called Freezing) 
    
    # Evaluate Strategy 1
    metrics = learn1.validate()
    frozen_acc = metrics[2]  # Accuracy değeri (3. eleman)
    results['frozen_head_only'] = frozen_acc
    print(f"Accuracy with frozen body, trained head: {frozen_acc:.4f}")
    learn1.save('transfer_strategy1')
    
    # Strategy 2: Progressive unfreezing (train final layer, then unfreeze gradually)
    print("\nStrategy 2: Progressive unfreezing")
    learn2 = vision_learner(dls, eval(model_arch), metrics=[error_rate_metric, accuracy_metric], pretrained=True)
    #B.3. Continue training all the weights F' and previous weights (called Unfreezing)
    # First train only the head
    learn2.fit_one_cycle(epochs, lr)
    
    # Then unfreeze all layers and train with discriminative learning rates
    learn2.unfreeze()
    
    # B.1. Learning Rate considerations:
    # B.1.1 Using too large learning rates: Convergence will be poor, if not impossible
    # B.1.2 Using too small learning rates: Convergence will be slow, takes too many epochs, risks overfitting
    
    # B.2. This implementation applies similar principles to Learning Rate Finder:
    # B.2.1 We determined a base learning rate (lr) through experimentation
    # B.2.2 For early layers (general features), we use a smaller lr (lr/100) to make minor tweaks
    # B.2.3 For later layers (specific features), we use a larger lr (lr/10) for more significant updates
    # B.2.4 This forms a "slice" of learning rates that increases from early to later layers
    # B.2.5 Similar to how Learning Rate Finder gradually increases lr to find optimal values
    learn2.fit_one_cycle(epochs, slice(lr/100, lr/10))
    
    # The slice(lr/100, lr/10) creates a range of learning rates:
    # - Early layers: lr/100 (very small adjustments to preserve general features)
    # - Middle layers: gradually increasing rates
    # - Later layers: lr/10 (larger adjustments to adapt specific features to our task)
    # This is an application of the ULMFIT approach for transfer learning
    
    # Evaluate Strategy 2
    metrics = learn2.validate()
    progressive_acc = metrics[2]  # Accuracy 
    results['progressive_unfreezing'] = progressive_acc
    print(f"Accuracy with progressive unfreezing: {progressive_acc:.4f}")
    learn2.save('transfer_strategy2')
    
    # Strategy 3: Fine-tuning (fastai's recommended approach)
    print("\nStrategy 3: Fine-tuning (fastai's approach)")
    learn3 = vision_learner(dls, eval(model_arch), metrics=[error_rate_metric, accuracy_metric], pretrained=True)
    
    # fine_tune automatically does: train head, unfreeze, train all
    learn3.fine_tune(epochs)
    
    # Evaluate Strategy 3
    metrics = learn3.validate()
    finetune_acc = metrics[2]  # Accuracy
    results['fine_tune'] = finetune_acc
    print(f"Accuracy with fine_tune: {finetune_acc:.4f}")
    learn3.save('transfer_strategy3')
    
    # Determine best strategy
    best_acc = max(results.values())
    best_strategy_name = [k for k, v in results.items() if v == best_acc][0]
    
    print("\nTransfer Learning Strategy Comparison:")
    for strategy, accuracy in results.items():
        print(f"{strategy}: {accuracy:.4f}")
    
    print(f"\nBest strategy: {best_strategy_name} with accuracy {best_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    strategies = list(results.keys())
    accuracies = [results[s] for s in strategies]
    
    plt.bar(strategies, accuracies)
    plt.ylim(0, 1.0)
    plt.xlabel('Transfer Learning Strategy')
    plt.ylabel('Validation Accuracy')
    plt.title('Comparison of Transfer Learning Strategies')
    plt.savefig('transfer_learning_comparison.png')
    plt.close()
    
    # Return the best learner
    if best_strategy_name == 'frozen_head_only':
        return learn1, results
    elif best_strategy_name == 'progressive_unfreezing':
        return learn2, results
    else:
        return learn3, results

# B.3-B.7 Advanced training with all optimizations
def train_optimized_model(dls, model_arch='resnet34', epochs=3, batch_size=32):
    """
    Trains a model with all optimizations including:
    - B.3 Transfer Learning
    - B.4 Discriminative Learning Rates
    - B.5 Optimal Number of Training Epochs
    - B.6 Model Capacity Adjustments
    - B.7 Proper Weight Initialization
    """
    print("=" * 50)
    print(f"TRAINING OPTIMIZED MODEL: {model_arch}")
    print("=" * 50)
    
    # B.6.1: When increasing the model capacity, ensure smaller batch size
    print(f"B.6.1: Using batch size {batch_size} for model {model_arch}")
    
    # Create learner with selected architecture (B.6 Model Capacity)
    # Transfer learning is automatically applied by using a pretrained model (B.3)
    # B.3: Replace the final linear layer F with a new one F' for our classification task
    print("B.3: Applying transfer learning - replacing final linear layer with new one for our task")
    learn = vision_learner(dls, 
                      eval(model_arch), 
                      metrics=[error_rate_metric, accuracy_metric],
                      pretrained=True)
    
    # B.6.3 Apply mixed precision training to optimize memory usage and speed 
    # This is similar to quantization in LLMs - using float16 instead of float32
    learn.to_fp16()
    print("B.6.3: Applied mixed precision training (float16) for better GPU memory usage")
    
    # B.2: Learning Rate Finder implementation
    # B.2.1-B.2.5: Finding optimal learning rate by starting with very low lr and gradually increasing
    print("B.2: Running learning rate finder to determine optimal learning rate")
    print("B.2.1-B.2.5: Starting with very low learning rate and gradually increasing")
    
    # Use built-in learning rate finder 
    lr_suggestions = find_learning_rate(learn)
    fastai_suggested_lr = lr_suggestions[0]  
    
    # Our manual learning rate finder implementation
    optimal_lr, lrs, losses = manual_lr_finder(learn, start_lr=1e-7, factor=3, max_iterations=15)
    
    print(f"Learning rate finder results:")
    print(f"- Fastai suggested learning rate: {fastai_suggested_lr}")
    print(f"- Our manual finder suggested: {optimal_lr}")
    
    # Select final learning rate based on finder results
    final_lr = fastai_suggested_lr
    print(f"Selected learning rate: {final_lr}")
    
    # B.4: Discriminative Learning Rates
    print("B.4: Applying Discriminative Learning Rates")
    print("B.4.1-B.4.2: Earlier layers need smaller learning rates, newer layers need larger tweaks")
    print("B.4.3-B.4.5: Earlier layers have more general features, later layers have more specific features")
    
    # B.5: Deciding the Number of Training Epochs
    print("B.5: Optimal epoch selection instead of early stopping")
    print("B.5.1-B.5.3: We don't use early stopping as it may counteract learning rate finder")
    
    # B.3: Applying best transfer learning strategy
    # First train only the new head (freezing pre-trained layers)
    print("B.3: Training only the new head first (Freezing)")
    learn.fit_one_cycle(1, final_lr)
    
    # Then unfreeze all layers and train with discriminative learning rates
    print("B.3: Continue training all weights (Unfreezing)")
    print("B.4.6-B.4.7: Using slice of learning rates - smaller for early layers, larger for later layers")
    
    # B.4.6-B.4.7: Apply discriminative learning rates using slice
    # Early layers get smaller learning rate (lr/100), later layers get higher (lr/10)
    learn.unfreeze()
    
    # B.1. Learning Rate considerations:
    # B.1.1 Using too large learning rates: Convergence will be poor, if not impossible
    # B.1.2 Using too small learning rates: Convergence will be slow, takes too many epochs, risks overfitting
    
    # B.2. This implementation applies principles derived from Learning Rate Finder:
    # B.2.1 We determined an optimal learning rate (final_lr) using learning rate finder
    # B.2.2 For early layers (general features), we use a smaller lr (final_lr/100) for minimal adjustments
    # B.2.3 For later layers (specific features), we use a larger lr (final_lr/10) for more significant updates
    # B.2.4 This forms a "slice" of learning rates that increases from early to later layers
    # B.2.5 Each layer gets an appropriate learning rate based on its depth in the network
    learn.fit_one_cycle(epochs-1, slice(final_lr/100, final_lr/10))
    
    # The slice(final_lr/100, final_lr/10) creates a range of learning rates:
    # - Early layers: final_lr/100 (very small adjustments to preserve general features)
    # - Middle layers: gradually increasing rates
    # - Later layers: final_lr/10 (larger adjustments to adapt specific features to our task)
    # This is an application of the ULMFIT approach for transfer learning, optimized based on our findings
    
    # Save final model
    learn.save('optimized_model_final')
    learn.export('optimized_emotion_classifier.pkl')
    
    # Evaluate final model
    print("Evaluating final optimized model...")
    interpret_model(learn, "optimized")
    
    return learn

# Full training pipeline
def main():
    # Set seed for reproducibility
    set_seed(42, reproducible=True)
    
    # Data path setup
    data_path = Path("EMOTION RECOGNITION DATASET")
    
    print("OPTIMIZED EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 50)
    
    # A.1 Data inspection
    classes, class_counts = inspect_dataset(data_path)
    
    # A.2 Create DataBlock with optimal parameters for main model
    result = create_datablock(
        data_path, 
        img_size=224,         # Standard size for most pretrained models
        batch_size=32,        # B.6.1: Adjusted smaller batch size for larger model capacity
        valid_pct=0.2         # 80/20 split
    )
    emotion_data = result[0]
    dls = result[1]
    
    # A.3 Inspect DataBlock
    class_names = inspect_datablock(emotion_data, dls, data_path)
    
    # STEP 1: A.4 Train and save benchmark model (ResNet18)
    print("\n=== STEP 1: TRAIN AND SAVE BENCHMARK MODEL (ResNet18) ===")
    benchmark_model = train_benchmark_model(dls, model_name='resnet18')
    
    # Benchmark model (ResNet18) metrics
    metrics = benchmark_model.validate()
    benchmark_valid_loss = metrics[0]  #  loss
    benchmark_accuracy = metrics[2]    #  accuracy (metrics = [loss, error_rate, accuracy])
    print(f"Benchmark Model (ResNet18) - Accuracy: {benchmark_accuracy:.4f}, Valid Loss: {benchmark_valid_loss:.4f}")
    benchmark_model.save('benchmark_resnet18_model')
    
    # STEP 2: Model comparison (ResNet18 vs ResNet34)
    # B.6: Model Capacity - comparing different model capacities
    print("\n=== STEP 2: MODEL COMPARISON (ResNet18 vs ResNet34) ===")
    # Train ResNet34 model
    print("Training ResNet34 model...")
    resnet34_model = train_benchmark_model(dls, model_name='resnet34')
    
    # ResNet34 metrics
    metrics = resnet34_model.validate()
    resnet34_valid_loss = metrics[0]  # loss
    resnet34_accuracy = metrics[2]    # accuracy
    print(f"ResNet34 Model - Accuracy: {resnet34_accuracy:.4f}, Valid Loss: {resnet34_valid_loss:.4f}")
    resnet34_model.save('benchmark_resnet34_model')
    
    # Compare both models and select the best one
    print("\n--- MODEL COMPARISON RESULTS ---")
    # B.6: Model Capacity Adjustments - Comparing models with different capacities (ResNet18 vs ResNet34)
    # This replaces the more general compare_model_capacities function with a specific implementation
    model_results = {
        'resnet18': {
            'accuracy': benchmark_accuracy,
            'valid_loss': benchmark_valid_loss,
            'model': benchmark_model
        },
        'resnet34': {
            'accuracy': resnet34_accuracy,
            'valid_loss': resnet34_valid_loss,
            'model': resnet34_model
        }
    }
    
    # Select model with highest accuracy
    best_model_arch = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
    best_accuracy = model_results[best_model_arch]['accuracy']
    best_model = model_results[best_model_arch]['model']
    
    print(f"Model Comparison Results:")
    for arch, metrics in model_results.items():
        print(f"{arch}: Accuracy={metrics['accuracy']:.4f}, Valid Loss={metrics['valid_loss']:.4f}")
    
    print(f"\nBest model: {best_model_arch} (accuracy: {best_accuracy:.4f})")
    
    # Apply B.6.1: When increasing model capacity, decrease batch size
    # This implements a key principle from compare_model_capacities function
    best_batch_size = 32 if best_model_arch == 'resnet34' else 64
    print(f"B.6.1: Using batch size {best_batch_size} for {best_model_arch}")
    
    # Create comparison graph
    plt.figure(figsize=(10, 6))
    archs = list(model_results.keys())
    accs = [model_results[arch]['accuracy'] for arch in archs]
    losses = [model_results[arch]['valid_loss'] for arch in archs]
    
    plt.subplot(1, 2, 1)
    plt.bar(archs, accs)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    
    plt.subplot(1, 2, 2)
    plt.bar(archs, losses)
    plt.title('Model Loss Comparison')
    
    plt.tight_layout()
    plt.savefig('model_architecture_comparison.png')
    plt.close()
    
    # STEP 3: Transfer Learning Strategies Comparison
    # B.3: Compare different transfer learning strategies
    print(f"\n=== STEP 3: TRANSFER LEARNING STRATEGIES COMPARISON ({best_model_arch}) ===")
    
    try:
        # Create new DataLoader for selected best model architecture
        result = create_datablock(
            data_path,
            img_size=224,
            batch_size=best_batch_size,  # B.6.1: Using appropriate batch size for model capacity
            valid_pct=0.2
        )
        transfer_dls = result[1]
        
        # B.3: Compare transfer learning strategies
        print("B.3: Comparing different transfer learning strategies:")
        print("1. Training only new head with frozen body")
        print("2. Progressive unfreezing")
        print("3. Fine-tuning (fastai approach)")
        
        best_strategy_model, strategy_results = compare_transfer_learning_strategies(
            transfer_dls,
            model_arch=best_model_arch,  # Use the best selected model architecture
            lr=1e-3,
            # B.5: Optimal Training Epochs - Using a fixed number instead of dynamically determining optimal epochs
            # This replaces the determine_optimal_epochs function with a simpler approach
            epochs=3  
        )
        
        print(f"\nTransfer Learning Strategies Results:")
        for strategy, accuracy in strategy_results.items():
            print(f"{strategy}: {accuracy:.4f}")
            
        # Determine best strategy
        best_strategy = max(strategy_results.keys(), key=lambda k: strategy_results[k])
        print(f"Best transfer strategy: {best_strategy}")
        
        # Transfer strategies comparison graph
        plt.figure(figsize=(10, 6))
        plt.bar(strategy_results.keys(), strategy_results.values())
        plt.title('Transfer Learning Strategies Comparison')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig('transfer_learning_comparison.png')
        plt.close()
        
    except Exception as e:
        print(f"Transfer learning strategy comparison error: {e}")
        print("Using default fine-tune strategy")
        best_strategy = "fine_tune"  # Default strategy
    
    # STEP 4: Final optimized model training
    # Applying all advanced techniques B.2-B.6
    print(f"\n=== STEP 4: FINAL OPTIMIZED MODEL TRAINING ===")
    print(f"Architecture: {best_model_arch}, Batch Size: {best_batch_size}, Strategy: {best_strategy}")
    
    try:
        # Create DataLoader for final model
        result = create_datablock(
            data_path,
            img_size=224,
            batch_size=best_batch_size, # B.6.1: Adjusted batch size for model capacity
            valid_pct=0.2
        )
        final_dls = result[1]
        
        # Train optimized model with all advanced techniques
        # B.2: Learning Rate Finder
        # B.3: Transfer Learning
        # B.4: Discriminative Learning Rates - Applied inside train_optimized_model using slice(lr/100, lr/10)
        # B.5: Optimal Epoch Selection - Using fixed epochs instead of dynamically determining optimal number
        # B.6: Model Capacity Adjustments - Using the best model architecture selected earlier
        print("Training final model with all optimizations:")
        print("- B.2: Learning Rate Finder")
        print("- B.3: Transfer Learning")
        print("- B.4: Discriminative Learning Rates")
        print("- B.5: Optimal Epoch Selection")
        print("- B.6: Model Capacity Adjustments")
        
        optimized_model = train_optimized_model(
            final_dls, 
            model_arch=best_model_arch,    # Selected best architecture
            # B.5: Using fixed epochs instead of dynamically determining optimal number
            epochs=3,                      # Final model epochs
            batch_size=best_batch_size     # B.6.1: Adjusted batch size according to model capacity
        )
        
        # STEP 5: Final model evaluation
        print("\n=== STEP 5: FINAL MODEL EVALUATION ===")
        metrics = optimized_model.validate()
        final_loss = metrics[0]     #  loss
        final_accuracy = metrics[2]  #  accuracy
        print(f"Final Optimized Model - Accuracy: {final_accuracy:.4f}, Loss: {final_loss:.4f}")
        
        # Confusion matrix and classification report
        interpret_model(optimized_model, "final_optimized")
        
        print("\nTraining completed successfully!")
        print(f"Model saved as: optimized_model_final")
        print(f"Exported as: optimized_emotion_classifier.pkl")
        
    except Exception as e:
        print(f"Final optimized model training error: {e}")
        print("Training could not be completed. Check error message.")

if __name__ == "__main__":
    main() 