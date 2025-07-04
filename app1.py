#Import Gradio for building the interactive web UI
import gradio as gr

# Import Hugging Face Transformers pipeline for sentiment analysis
from transformers import pipeline

# Import matplotlib to draw the bar chart for sentiment summary
import matplotlib.pyplot as plt

#tf.compat.v1.losses.sparse_softmax_cross_entropy instead.


# 🔍 Load the pre-trained sentiment analysis model from local directory
model = pipeline("text-classification", model="sentiment_model")

# ✨ Define the main function to analyze a batch of texts
def analyze_texts(texts):
    results = model(texts)  # Run the model on the list of input texts
    
    labels = []       # Stores predicted labels (POSITIVE/NEGATIVE)
    confidences = []  # Stores prediction confidence scores
    emojis = []       # Stores decorated output for the user
    
    # 🎯 Iterate through each model output and extract relevant info
    for res in results:
        label = res['label']         # Sentiment label
        score = res['score']         # Confidence score
        emoji = "🟢" if label == "POSITIVE" else "🔴"  # Choose emoji
        labels.append(label)
        confidences.append(score)
        # Create a styled prediction string with emoji + confidence
        emojis.append(f"{emoji} **{label}** ({score:.2%})")

    # 📊 Generate a sentiment distribution bar chart
    fig, ax = plt.subplots(figsize=(4, 3))  # Create figure and axis
    label_counts = {
        "POSITIVE": labels.count("POSITIVE"),
        "NEGATIVE": labels.count("NEGATIVE")
    }  # Count how many positive/negative predictions
    ax.bar(label_counts.keys(), label_counts.values(), color=["green", "red"])  # Draw bars
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(label_counts.values()) + 1)  # Add headroom above bars
    plt.tight_layout()  # Fix layout spacing

    return emojis, fig  # Return predictions + chart to Gradio

# 🎨 UI configuration
title = "🧠 Sentiment Analyzer Pro"  # App title
description = "Paste multiple reviews (one per line) to see predictions + a sentiment chart."  # App description

# 📝 Example input text for the demo
examples = [[
    "I love this phone!\nTerrible customer service.\nPretty average overall."
]]

# 🧱 Build the UI using Gradio Blocks layout with dark theme

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.Markdown(f"# {title}")         # Show title as a header
    gr.Markdown(description)          # Show description text

    # ✍️ Text input box for multiple reviews (1 per line)
    input_box = gr.Textbox(
        label="💬 Enter multiple reviews (one per line)", 
        lines=6,
        placeholder="E.g.\nThis is great!\nNot what I expected.\nIt’s okay."
    )

    # 📣 Output section for styled predictions
    output_preds = gr.Markdown(label="📣 Predictions with Emojis")

    # 📊 Output section for the bar chart
    output_chart = gr.Plot(label="📊 Sentiment Bar Chart")

    # 🔘 Button that triggers the analysis
    analyze_btn = gr.Button("🔍 Analyze")

    # 🔗 Connect the button to the analysis function
    analyze_btn.click(
        fn=lambda text: analyze_texts(text.strip().split("\n")),  # Split input by lines
        inputs=input_box,
        outputs=[output_preds, output_chart]
    )

# 🚀 Launch the app
demo.launch()
