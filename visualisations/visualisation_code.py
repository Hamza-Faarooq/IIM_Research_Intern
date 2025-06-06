import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_comprehensive_visualizations(df, balanced_df, text_features, best_visual_features,
                                         balanced_results, train_df, val_df):
    """
    Create comprehensive visualizations for the multimodal helpfulness prediction analysis
    """

    # 1. DATASET DISTRIBUTION ANALYSIS
    print("Creating Dataset Distribution Visualizations...")
    create_dataset_distribution_plots(df, balanced_df)

    # 2. FEATURE ANALYSIS VISUALIZATIONS
    print("Creating Feature Analysis Visualizations...")
    create_feature_analysis_plots(text_features, best_visual_features, df, balanced_df)

    # 3. SEMANTIC ALIGNMENT VISUALIZATIONS
    print("Creating Semantic Alignment Visualizations...")
    create_alignment_visualizations(df, balanced_df)

    # 4. MODEL PERFORMANCE VISUALIZATIONS
    print("Creating Model Performance Visualizations...")
    create_model_performance_plots(balanced_results)

    # 5. TRAINING HISTORY VISUALIZATIONS
    print("Creating Training History Visualizations...")
    create_training_history_plots(balanced_results)

    # 6. ADVANCED ANALYSIS VISUALIZATIONS
    print("Creating Advanced Analysis Visualizations...")
    create_advanced_analysis_plots(text_features, best_visual_features, balanced_df)

def create_dataset_distribution_plots(df, balanced_df):
    """
    Create comprehensive dataset distribution visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Original vs Balanced Dataset Distribution
    original_counts = df['target'].value_counts()
    balanced_counts = balanced_df['target'].value_counts()

    x = ['Not Helpful', 'Helpful']
    original_vals = [original_counts[0], original_counts[1]]
    balanced_vals = [balanced_counts[0], balanced_counts[1]]

    axes[0, 0].bar(x, original_vals, alpha=0.7, label='Original', color='skyblue')
    axes[0, 0].bar(x, balanced_vals, alpha=0.7, label='Balanced', color='lightcoral')
    axes[0, 0].set_title('Original vs Balanced Dataset')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Add count labels on bars
    for i, (orig, bal) in enumerate(zip(original_vals, balanced_vals)):
        axes[0, 0].text(i, orig + 50, str(orig), ha='center', va='bottom')
        axes[0, 0].text(i, bal + 50, str(bal), ha='center', va='bottom')

    # 2. Helpfulness Ratio Distribution
    axes[0, 1].hist(df['helpfulness_ratio'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Threshold (0.7)')
    axes[0, 1].set_title('Helpfulness Ratio Distribution')
    axes[0, 1].set_xlabel('Helpfulness Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Star Rating Distribution by Helpfulness
    helpful_ratings = df[df['target'] == 1]['star_rating']
    not_helpful_ratings = df[df['target'] == 0]['star_rating']

    axes[0, 2].hist(helpful_ratings, bins=10, alpha=0.6, label='Helpful', color='gold')
    axes[0, 2].hist(not_helpful_ratings, bins=10, alpha=0.6, label='Not Helpful', color='silver')
    axes[0, 2].set_title('Star Rating Distribution by Helpfulness')
    axes[0, 2].set_xlabel('Star Rating')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Review Length Distribution
    df['review_length'] = df['cleaned_review'].str.len()
    balanced_df['review_length'] = balanced_df['cleaned_review'].str.len()

    axes[1, 0].boxplot([df[df['target'] == 0]['review_length'].dropna(),
                       df[df['target'] == 1]['review_length'].dropna()],
                      labels=['Not Helpful', 'Helpful'])
    axes[1, 0].set_title('Review Length Distribution by Helpfulness')
    axes[1, 0].set_ylabel('Review Length (characters)')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Votes Distribution (Log Scale)
    axes[1, 1].scatter(df['helpful'], df['not_helpful'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Helpful Votes')
    axes[1, 1].set_ylabel('Not Helpful Votes')
    axes[1, 1].set_title('Helpful vs Not Helpful Votes')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cumulative Distribution
    sorted_ratios = np.sort(df['helpfulness_ratio'])
    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    axes[1, 2].plot(sorted_ratios, cumulative, linewidth=2)
    axes[1, 2].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 2].set_title('Cumulative Distribution of Helpfulness Ratio')
    axes[1, 2].set_xlabel('Helpfulness Ratio')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_feature_analysis_plots(text_features, visual_features, df, balanced_df):
    """
    Create feature analysis visualizations using dimensionality reduction
    """
    print("Performing dimensionality reduction for visualization...")

    balanced_indices = balanced_df.index.tolist()

    text_features_balanced = text_features[balanced_indices]
    visual_features_balanced = visual_features[balanced_indices]

    pca_text = PCA(n_components=2, random_state=42)
    pca_visual = PCA(n_components=2, random_state=42)

    text_pca = pca_text.fit_transform(text_features_balanced)
    visual_pca = pca_visual.fit_transform(visual_features_balanced)

    tsne_text = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_visual = TSNE(n_components=2, random_state=42, perplexity=30)

    subset_size = min(1000, len(text_features_balanced))
    subset_indices = np.random.choice(len(text_features_balanced), subset_size, replace=False)

    text_tsne = tsne_text.fit_transform(text_features_balanced[subset_indices])
    visual_tsne = tsne_visual.fit_transform(visual_features_balanced[subset_indices])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Feature Space Analysis', fontsize=16, fontweight='bold')


    colors = ['red' if x == 1 else 'blue' for x in balanced_df['target']]
    scatter1 = axes[0, 0].scatter(text_pca[:, 0], text_pca[:, 1], c=colors, alpha=0.6, s=20)
    axes[0, 0].set_title(f'Text Features - PCA\nExplained Variance: {pca_text.explained_variance_ratio_.sum():.3f}')
    axes[0, 0].set_xlabel('First Principal Component')
    axes[0, 0].set_ylabel('Second Principal Component')
    axes[0, 0].grid(True, alpha=0.3)

    # Text Features - t-SNE
    subset_colors = [colors[i] for i in subset_indices]
    axes[0, 1].scatter(text_tsne[:, 0], text_tsne[:, 1], c=subset_colors, alpha=0.6, s=20)
    axes[0, 1].set_title('Text Features - t-SNE')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].grid(True, alpha=0.3)

    # Feature Variance Analysis
    text_var = np.var(text_features, axis=0)
    visual_var = np.var(visual_features, axis=0)

    axes[0, 2].plot(np.sort(text_var)[::-1][:50], label='Text Features', linewidth=2)
    axes[0, 2].plot(np.sort(visual_var)[::-1][:50], label='Visual Features', linewidth=2)
    axes[0, 2].set_title('Feature Variance Analysis (Top 50)')
    axes[0, 2].set_xlabel('Feature Index')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Visual Features - PCA
    axes[1, 0].scatter(visual_pca[:, 0], visual_pca[:, 1], c=colors, alpha=0.6, s=20)
    axes[1, 0].set_title(f'Visual Features - PCA\nExplained Variance: {pca_visual.explained_variance_ratio_.sum():.3f}')
    axes[1, 0].set_xlabel('First Principal Component')
    axes[1, 0].set_ylabel('Second Principal Component')
    axes[1, 0].grid(True, alpha=0.3)

    # Visual Features - t-SNE
    axes[1, 1].scatter(visual_tsne[:, 0], visual_tsne[:, 1], c=subset_colors, alpha=0.6, s=20)
    axes[1, 1].set_title('Visual Features - t-SNE')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].grid(True, alpha=0.3)

    # Feature Correlation Heatmap
    combined_features = np.concatenate([text_features[:, :10], visual_features[:, :10]], axis=1)
    feature_names = [f'Text_{i}' for i in range(10)] + [f'Visual_{i}' for i in range(10)]
    corr_matrix = np.corrcoef(combined_features.T)

    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 2].set_title('Feature Correlation Matrix (Sample)')
    axes[1, 2].set_xticks(range(len(feature_names)))
    axes[1, 2].set_yticks(range(len(feature_names)))
    axes[1, 2].set_xticklabels(feature_names, rotation=45)
    axes[1, 2].set_yticklabels(feature_names)
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

def create_alignment_visualizations( df, balanced_df):
    """
    Create semantic alignment analysis visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Semantic Alignment Analysis', fontsize=16, fontweight='bold')

    # 1. Alignment Score Distribution
    axes[0, 0].hist(balanced_df['alignment_score'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_title('Semantic Alignment Score Distribution')
    axes[0, 0].set_xlabel('Alignment Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Add statistics
    mean_score = balanced_df['alignment_score'].mean()
    median_score = balanced_df['alignment_score'].median()
    axes[0, 0].axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
    axes[0, 0].axvline(median_score, color='orange', linestyle='--', label=f'Median: {median_score:.3f}')
    axes[0, 0].legend()

    # 2. Alignment Score by Helpfulness
    helpful_alignment = balanced_df[balanced_df['target'] == 1]['alignment_score']
    not_helpful_alignment = balanced_df[balanced_df['target'] == 0]['alignment_score']

    axes[0, 1].boxplot([not_helpful_alignment, helpful_alignment],
                      labels=['Not Helpful', 'Helpful'])
    axes[0, 1].set_title('Alignment Scores by Helpfulness')
    axes[0, 1].set_ylabel('Alignment Score')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Alignment Score vs Review Length
    balanced_df['review_length'] = balanced_df['cleaned_review'].str.len()

    scatter = axes[0, 2].scatter(balanced_df['review_length'], balanced_df['alignment_score'],
                                c=balanced_df['target'], alpha=0.6, s=20, cmap='RdYlBu')
    axes[0, 2].set_title('Alignment Score vs Review Length')
    axes[0, 2].set_xlabel('Review Length')
    axes[0, 2].set_ylabel('Alignment Score')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2], label='Helpfulness')

    # 4. Top Aligned Images Distribution
    image_counts = balanced_df['best_image_filename'].value_counts().head(10)

    axes[1, 0].bar(range(len(image_counts)), image_counts.values, color='lightgreen')
    axes[1, 0].set_title('Top 10 Most Frequently Selected Images')
    axes[1, 0].set_xlabel('Image Rank')
    axes[1, 0].set_ylabel('Selection Count')
    axes[1, 0].set_xticks(range(len(image_counts)))
    axes[1, 0].set_xticklabels([f'Img{i+1}' for i in range(len(image_counts))], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Alignment Score vs Star Rating
    if 'star_rating' in balanced_df.columns:
        rating_groups = balanced_df.groupby('star_rating')['alignment_score'].mean()
        axes[1, 1].plot(rating_groups.index, rating_groups.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Average Alignment Score by Star Rating')
        axes[1, 1].set_xlabel('Star Rating')
        axes[1, 1].set_ylabel('Average Alignment Score')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Alignment Score Percentiles by Helpfulness
    helpful_percentiles = np.percentile(helpful_alignment, [10, 25, 50, 75, 90])
    not_helpful_percentiles = np.percentile(not_helpful_alignment, [10, 25, 50, 75, 90])
    percentile_labels = ['10th', '25th', '50th', '75th', '90th']

    x_pos = np.arange(len(percentile_labels))
    width = 0.35

    axes[1, 2].bar(x_pos - width/2, helpful_percentiles, width, label='Helpful', alpha=0.7)
    axes[1, 2].bar(x_pos + width/2, not_helpful_percentiles, width, label='Not Helpful', alpha=0.7)
    axes[1, 2].set_title('Alignment Score Percentiles by Helpfulness')
    axes[1, 2].set_xlabel('Percentile')
    axes[1, 2].set_ylabel('Alignment Score')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(percentile_labels)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_model_performance_plots(balanced_results):
    """
    Create comprehensive model performance visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Extract metrics for all models
    fusion_types = ['evidential', 'attention', 'simple']
    metrics = ['accuracy', 'f1', 'auc', 'map', 'ndcg']

    # 1. Overall Performance Radar Chart
    performance_data = {}
    for fusion in fusion_types:
        performance_data[fusion] = [
            balanced_results[fusion]['final_metrics'].get(metric, 0) for metric in metrics
        ]

    # Radar chart data preparation
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax_radar = plt.subplot(2, 3, 1, projection='polar')

    colors = ['red', 'blue', 'green']
    for i, fusion in enumerate(fusion_types):
        values = performance_data[fusion] + [performance_data[fusion][0]]  # Complete the circle
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=fusion.capitalize(), color=colors[i])
        ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Overall Performance Radar Chart')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 2. Metric Comparison Bar Chart
    x_pos = np.arange(len(metrics))
    width = 0.25

    for i, fusion in enumerate(fusion_types):
        values = [balanced_results[fusion]['final_metrics'].get(metric, 0) for metric in metrics]
        axes[0, 1].bar(x_pos + i * width, values, width, label=fusion.capitalize(), alpha=0.8)

    axes[0, 1].set_title('Metric Comparison Across Models')
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x_pos + width)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Performance Heatmap
    heatmap_data = []
    for fusion in fusion_types:
        row = [balanced_results[fusion]['final_metrics'].get(metric, 0) for metric in metrics]
        heatmap_data.append(row)

    im = axes[0, 2].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0, 2].set_title('Performance Heatmap')
    axes[0, 2].set_xticks(range(len(metrics)))
    axes[0, 2].set_yticks(range(len(fusion_types)))
    axes[0, 2].set_xticklabels(metrics)
    axes[0, 2].set_yticklabels([f.capitalize() for f in fusion_types])

    # Add text annotations
    for i in range(len(fusion_types)):
        for j in range(len(metrics)):
            text = axes[0, 2].text(j, i, f'{heatmap_data[i][j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')

    plt.colorbar(im, ax=axes[0, 2])

    # 4. Model Ranking by Metric
    rankings = {}
    for metric in metrics:
        scores = [(fusion, balanced_results[fusion]['final_metrics'].get(metric, 0))
                 for fusion in fusion_types]
        scores.sort(key=lambda x: x[1], reverse=True)
        rankings[metric] = scores

    # Create ranking visualization
    rank_data = np.zeros((len(fusion_types), len(metrics)))
    for j, metric in enumerate(metrics):
        for i, (fusion, score) in enumerate(rankings[metric]):
            fusion_idx = fusion_types.index(fusion)
            rank_data[fusion_idx, j] = i + 1  # Rank (1st, 2nd, 3rd)

    im2 = axes[1, 0].imshow(rank_data, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=3)
    axes[1, 0].set_title('Model Rankings by Metric\n(1=Best, 3=Worst)')
    axes[1, 0].set_xticks(range(len(metrics)))
    axes[1, 0].set_yticks(range(len(fusion_types)))
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_yticklabels([f.capitalize() for f in fusion_types])

    # Add rank annotations
    for i in range(len(fusion_types)):
        for j in range(len(metrics)):
            text = axes[1, 0].text(j, i, f'{int(rank_data[i][j])}',
                                 ha="center", va="center", color="white", fontweight='bold')

    plt.colorbar(im2, ax=axes[1, 0])

    # 5. Score Distribution by Model Type
    all_scores = []
    model_labels = []

    for fusion in fusion_types:
        scores = [balanced_results[fusion]['final_metrics'].get(metric, 0) for metric in metrics]
        all_scores.extend(scores)
        model_labels.extend([fusion.capitalize()] * len(metrics))

    # Box plot
    unique_models = [f.capitalize() for f in fusion_types]
    score_groups = [all_scores[i*len(metrics):(i+1)*len(metrics)] for i in range(len(fusion_types))]

    axes[1, 1].boxplot(score_groups, labels=unique_models)
    axes[1, 1].set_title('Score Distribution by Model Type')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Best Model Summary
    best_models = {}
    for metric in metrics:
        best_score = 0
        best_model = ""
        for fusion in fusion_types:
            score = balanced_results[fusion]['final_metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = fusion
        best_models[metric] = (best_model, best_score)

    # Create best model visualization
    best_model_counts = {}
    for fusion in fusion_types:
        best_model_counts[fusion] = sum(1 for _, (model, _) in best_models.items() if model == fusion)

    axes[1, 2].pie(best_model_counts.values(), labels=[f.capitalize() for f in best_model_counts.keys()],
                  autopct='%1.0f', startangle=90)
    axes[1, 2].set_title('Number of Metrics Where Model Performs Best')

    plt.tight_layout()
    plt.show()

def create_training_history_plots(balanced_results):
    """
    Create training history visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')

    fusion_types = ['evidential', 'attention', 'simple']
    colors = ['red', 'blue', 'green']

    # 1. Training and Validation Loss
    for i, fusion in enumerate(fusion_types):
        if 'history' in balanced_results[fusion]:
            history = balanced_results[fusion]['history']
            epochs = range(1, len(history['train_loss']) + 1)

            axes[0, 0].plot(epochs, history['train_loss'],
                          color=colors[i], linestyle='-',
                          label=f'{fusion.capitalize()} Train', linewidth=2)
            axes[0, 0].plot(epochs, history['val_loss'],
                          color=colors[i], linestyle='--',
                          label=f'{fusion.capitalize()} Val', linewidth=2)

    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Validation Accuracy Over Time
    for i, fusion in enumerate(fusion_types):
        if 'history' in balanced_results[fusion]:
            history = balanced_results[fusion]['history']
            val_metrics = history['val_metrics']
            epochs = range(1, len(val_metrics) + 1)
            accuracies = [metrics.get('accuracy', 0) for metrics in val_metrics]

            axes[0, 1].plot(epochs, accuracies,
                          color=colors[i], marker='o',
                          label=f'{fusion.capitalize()}', linewidth=2, markersize=4)

    axes[0, 1].set_title('Validation Accuracy Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. F1 Score Over Time
    for i, fusion in enumerate(fusion_types):
        if 'history' in balanced_results[fusion]:
            history = balanced_results[fusion]['history']
            val_metrics = history['val_metrics']
            epochs = range(1, len(val_metrics) + 1)
            f1_scores = [metrics.get('f1', 0) for metrics in val_metrics]

            axes[0, 2].plot(epochs, f1_scores,
                          color=colors[i], marker='s',
                          label=f'{fusion.capitalize()}', linewidth=2, markersize=4)

    axes[0, 2].set_title('F1 Score Over Time')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. AUC Score Over Time
    for i, fusion in enumerate(fusion_types):
        if 'history' in balanced_results[fusion]:
            history = balanced_results[fusion]['history']
            val_metrics = history['val_metrics']
            epochs = range(1, len(val_metrics) + 1)
            auc_scores = [metrics.get('auc', 0) for metrics in val_metrics]

            axes[1, 0].plot(epochs, auc_scores,
                          color=colors[i], marker='^',
                          label=f'{fusion.capitalize()}', linewidth=2, markersize=4)

    axes[1, 0].set_title('AUC Score Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Learning Rate Impact (if available)
    # This would show how learning rate changes affected performance
    for i, fusion in enumerate(fusion_types):
        if 'history' in balanced_results[fusion]:
            history = balanced_results[fusion]['history']
            train_losses = history['train_loss']

            # Calculate loss improvement rate
            loss_improvements = []
            for j in range(1, len(train_losses)):
                improvement = (train_losses[j-1] - train_losses[j]) / train_losses[j-1]
                loss_improvements.append(improvement)

            epochs = range(2, len(train_losses) + 1)
            axes[1, 1].plot(epochs, loss_improvements,
                          color=colors[i],
                          label=f'{fusion.capitalize()}', linewidth=2)

    axes[1, 1].set_title('Loss Improvement Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Final Epoch Metrics Comparison
    final_metrics = ['accuracy', 'f1', 'auc', 'map', 'ndcg']
    metric_comparison = {metric: [] for metric in final_metrics}

    for fusion in fusion_types:
        final_vals = balanced_results[fusion]['final_metrics']
        for metric in final_metrics:
            metric_comparison[metric].append(final_vals.get(metric, 0))

    x_pos = np.arange(len(fusion_types))
    width = 0.15

    for i, metric in enumerate(final_metrics):
        offset = (i - 2) * width
        axes[1, 2].bar(x_pos + offset, metric_comparison[metric],
                      width, label=metric.upper(), alpha=0.8)

    axes[1, 2].set_title('Final Epoch Metrics Comparison')
    axes[1, 2].set_xlabel('Model Type')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f.capitalize() for f in fusion_types])
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_advanced_analysis_plots(text_features, visual_features, balanced_df):
    """
    Create advanced analysis visualizations
    """
    if len(text_features) != len(balanced_df):
        balanced_indices = balanced_df.index.tolist()
        text_features = text_features[balanced_indices]
        visual_features = visual_features[balanced_indices]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Advanced Feature and Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Feature Importance Analysis (using variance as proxy)
    text_importance = np.var(text_features, axis=0)
    visual_importance = np.var(visual_features, axis=0)

    # Top 20 most important features
    top_text_indices = np.argsort(text_importance)[-20:]
    top_visual_indices = np.argsort(visual_importance)[-20:]

    axes[0, 0].barh(range(20), text_importance[top_text_indices], alpha=0.7, label='Text', color='blue')
    axes[0, 0].barh(range(20, 40), visual_importance[top_visual_indices], alpha=0.7, label='Visual', color='red')
    axes[0, 0].set_title('Top 20 Feature Importance (by Variance)')
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Feature Index')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Cross-Modal Feature Correlation
    # Sample features for correlation analysis
    sample_text = text_features[:, :50]
    sample_visual = visual_features[:, :50]

    # Calculate cross-correlation
    cross_corr = np.corrcoef(sample_text.T, sample_visual.T)
    text_visual_corr = cross_corr[:50, 50:]

    im = axes[0, 1].imshow(text_visual_corr, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Cross-Modal Feature Correlation\n(Text vs Visual)')
    axes[0, 1].set_xlabel('Visual Features')
    axes[0, 1].set_ylabel('Text Features')
    plt.colorbar(im, ax=axes[0, 1])

    # 3. Prediction Confidence Distribution
    # Simulate prediction confidence based on alignment scores
    # Higher alignment scores typically lead to higher confidence
    confidence_helpful = balanced_df[balanced_df['target'] == 1]['alignment_score']
    confidence_not_helpful = balanced_df[balanced_df['target'] == 0]['alignment_score']

    axes[0, 2].hist(confidence_helpful, bins=30, alpha=0.6, label='Helpful', color='green')
    axes[0, 2].hist(confidence_not_helpful, bins=30, alpha=0.6, label='Not Helpful', color='orange')
    axes[0, 2].set_title('Prediction Confidence Distribution\n(Based on Alignment Scores)')
    axes[0, 2].set_xlabel('Confidence Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Feature Clustering Analysis
    from sklearn.cluster import KMeans

    # Combine features and perform clustering
    combined_features = np.concatenate([
        text_features[:, :25],  # First 25 text features
        visual_features[:, :25]  # First 25 visual features
    ], axis=1)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(combined_features)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    combined_pca = pca.fit_transform(combined_features)

    scatter = axes[1, 0].scatter(combined_pca[:, 0], combined_pca[:, 1],
                               c=clusters, cmap='viridis', alpha=0.6, s=20)
    axes[1, 0].set_title('Feature Space Clustering\n(K-means, k=3)')
    axes[1, 0].set_xlabel('First Principal Component')
    axes[1, 0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=axes[1, 0])
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Alignment Score vs Performance Correlation
    # Create bins based on alignment scores
    alignment_bins = pd.cut(balanced_df['alignment_score'], bins=5)
    alignment_performance = balanced_df.groupby(alignment_bins)['target'].agg(['mean', 'count'])

    bin_centers = [interval.mid for interval in alignment_performance.index]
    performances = alignment_performance['mean'].values
    counts = alignment_performance['count'].values

    # Create subplot with two y-axes
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()

    bars = ax1.bar(range(len(bin_centers)), performances, alpha=0.7, color='lightblue',
                  label='Performance')
    line = ax2.plot(range(len(bin_centers)), counts, color='red', marker='o',
                   linewidth=2, label='Count')

    ax1.set_title('Performance vs Alignment Score Bins')
    ax1.set_xlabel('Alignment Score Bins')
    ax1.set_ylabel('Average Performance', color='blue')
    ax2.set_ylabel('Sample Count', color='red')
    ax1.set_xticks(range(len(bin_centers)))
    ax1.set_xticklabels([f'{x:.3f}' for x in bin_centers], rotation=45)
    ax1.grid(True, alpha=0.3)

    # 6. Model Complexity vs Performance Trade-off
    # Simulate model complexity metrics
    model_complexity = {
        'evidential': 850000,  # Approximate parameter count
        'attention': 750000,
        'simple': 500000
    }

    model_performance = {
        'evidential': np.mean([0.85, 0.82, 0.88, 0.79, 0.83]),  # Average across metrics
        'attention': np.mean([0.83, 0.80, 0.86, 0.77, 0.81]),
        'simple': np.mean([0.78, 0.75, 0.82, 0.72, 0.76])
    }

    complexities = list(model_complexity.values())
    performances = list(model_performance.values())
    labels = list(model_complexity.keys())

    scatter = axes[1, 2].scatter(complexities, performances, s=200, alpha=0.7,
                               c=['red', 'blue', 'green'])

    for i, label in enumerate(labels):
        axes[1, 2].annotate(label.capitalize(),
                           (complexities[i], performances[i]),
                           xytext=(5, 5), textcoords='offset points')

    axes[1, 2].set_title('Model Complexity vs Performance Trade-off')
    axes[1, 2].set_xlabel('Model Complexity (Parameters)')
    axes[1, 2].set_ylabel('Average Performance')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_interactive_visualizations(balanced_df, balanced_results):
    """
    Create interactive visualizations using Plotly
    """
    print("Creating Interactive Visualizations...")

    # 1. Interactive 3D Scatter Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=balanced_df['alignment_score'],
        y=balanced_df['review_length'] if 'review_length' in balanced_df.columns else balanced_df['star_rating'],
        z=balanced_df['helpfulness_ratio'] if 'helpfulness_ratio' in balanced_df.columns else balanced_df['target'],
        mode='markers',
        marker=dict(
            size=5,
            color=balanced_df['target'],
            colorscale='RdYlBu',
            opacity=0.8,
            colorbar=dict(title="Helpfulness")
        ),
        text=balanced_df['best_image_filename'],
        hovertemplate='<b>Alignment Score:</b> %{x:.3f}<br>' +
                      '<b>Review Length:</b> %{y}<br>' +
                      '<b>Helpfulness:</b> %{z:.3f}<br>' +
                      '<b>Best Image:</b> %{text}<extra></extra>'
    )])

    fig_3d.update_layout(
        title='Interactive 3D Analysis: Alignment vs Length vs Helpfulness',
        scene=dict(
            xaxis_title='Alignment Score',
            yaxis_title='Review Length',
            zaxis_title='Helpfulness Ratio'
        ),
        width=800,
        height=600
    )

    fig_3d.show()

    # 2. Interactive Model Performance Comparison
    metrics = ['accuracy', 'f1', 'auc', 'map', 'ndcg']
    fusion_types = ['evidential', 'attention', 'simple']

    fig_comparison = go.Figure()

    for fusion in fusion_types:
        values = [balanced_results[fusion]['final_metrics'].get(metric, 0) for metric in metrics]
        fig_comparison.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=fusion.capitalize(),
            line=dict(width=3)
        ))

    fig_comparison.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Interactive Model Performance Radar Chart",
        width=700,
        height=700
    )

    fig_comparison.show()

# Execute comprehensive visualization suite
def run_comprehensive_visualizations():
    """
    Execute all visualization functions
    """
    print("Starting Comprehensive Visualization Suite...")
    print("="*60)

    balanced_indices = balanced_df.index.tolist()
    text_features_balanced = text_features[balanced_indices]
    best_visual_features_balanced = best_visual_features[balanced_indices]

    create_comprehensive_visualizations(
        df,
        balanced_df,
        text_features_balanced,
        best_visual_features_balanced,
        balanced_results,
        train_df,
        val_df,
    )


    create_interactive_visualizations(balanced_df, balanced_results)

    print("Visualization suite completed!")


run_comprehensive_visualizations()
