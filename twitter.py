import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Define column names
columns = ['id', 'topic', 'original_sentiment', 'tweet']

# 2. Load datasets
train_data = pd.read_csv(r'C:\Users\Mokitha\OneDrive\Desktop\Data Science\Task 3\twitter_training.csv', names=columns)
validation_data = pd.read_csv(r'C:\Users\Mokitha\OneDrive\Desktop\Data Science\Task 3\twitter_validation.csv', names=columns)

# 3. Merge datasets
data = pd.concat([train_data, validation_data], ignore_index=True)

# 4. FAST TEST: Use only first 250 rows
data = data.head(250)

# 5. Sentiment prediction function
def get_sentiment(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    else:
        return 'Neutral'

# 6. Apply sentiment analysis
data['Predicted_Sentiment'] = data['tweet'].apply(get_sentiment)

# 7. Visualization - Pie Chart
plt.figure(figsize=(6,6))
data['Predicted_Sentiment'].value_counts().plot.pie(
    autopct='%1.0f%%', colors=['green', 'blue', 'red'], startangle=90, shadow=True
)
plt.title('Overall Sentiment Distribution')
plt.ylabel('')
plt.show()

# 8. WordCloud for Positive Tweets
positive_words = ' '.join(data[data['Predicted_Sentiment'] == 'Positive']['tweet'].dropna())
wordcloud_pos = WordCloud(background_color='white', width=800, height=400).generate(positive_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Tweets WordCloud')
plt.show()

# 9. WordCloud for Negative Tweets
negative_words = ' '.join(data[data['Predicted_Sentiment'] == 'Negative']['tweet'].dropna())
wordcloud_neg = WordCloud(background_color='black', colormap='Reds', width=800, height=400).generate(negative_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Tweets WordCloud')
plt.show()

# 10. Compare Original vs Predicted Sentiment (Optional)
compare = pd.DataFrame({
    'Original': data['original_sentiment'],
    'Predicted': data['Predicted_Sentiment']
})

compare.groupby(['Original', 'Predicted']).size().unstack(fill_value=0).plot(
    kind='bar', stacked=True, figsize=(10,6), colormap='viridis'
)
plt.title('Original Sentiment vs Predicted Sentiment')
plt.xlabel('Original Sentiment')
plt.ylabel('Number of Tweets')
plt.legend(title='Predicted Sentiment')
plt.xticks(rotation=45)
plt.show()

# Print the sentiment counts
print("\nSentiment Counts:")
print(data['Predicted_Sentiment'].value_counts())

# Show few example tweets
print("\nSample Tweets with Predicted Sentiments:")
print(data[['tweet', 'Predicted_Sentiment']].head(10))

# Calculate and print match accuracy
match = (data['original_sentiment'] == data['Predicted_Sentiment']).sum()
total = len(data)
accuracy = (match / total) * 100
print(f"\nAccuracy between Original and Predicted Sentiment: {accuracy:.2f}%")

plt.figure(figsize=(6,6))
data['Predicted_Sentiment'].value_counts().plot.pie(
    autopct='%1.0f%%', colors=['green', 'blue', 'red'], startangle=90, shadow=True
)
plt.title('Overall Sentiment Distribution')
plt.ylabel('')
plt.show(block=False)
plt.pause(3)
plt.close()
