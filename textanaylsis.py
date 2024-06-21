from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# load in pandas and numpy for data manipulation
import pandas as pd
import numpy as np




# Sample piece of text
sample_text = "still have depression symptoms not as bad as they used to be in fact my therapist says im improving a lot but for the past years ive been stuck in this state of emotional numbness feeling disconnected from myself others and the world and time doesnt seem to be passing"

# Run the model on the sample text
result=classifier(sample_text, text_labels, multi_label = False)
print(result)



def predict_sentiment(df, text_column, text_labels):
    

    result_list = []
    for index, row in df.iterrows():
        sequence_to_classify = row[text_column]
        result = classifier(sequence_to_classify, text_labels, multi_label = False)
        print(result)
        result['sentiment'] = result['labels'][0]
        result['score'] = result['scores'][0]
        result_list.append(result)
    result_df = pd.DataFrame(result_list)[['sequence','sentiment', 'score']]
    result_df = pd.merge(df, result_df, left_on = "text", right_on="sequence", how = "left")
    return result_df

df = pd.read_csv("small.csv")

results_df = predict_sentiment(df=df, text_column ="text", text_labels= text_labels)

csv_file_path = 'output.csv'
df.to_csv(csv_file_path, index=False)
print(f"DataFrame written to {csv_file_path}")
