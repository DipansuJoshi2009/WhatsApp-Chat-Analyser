# Creating the WhatsApp Chat Analyser
import streamlit as st
import pandas as pd
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import seaborn as sns
import emoji
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('omw-1.4')

# st.title('WhatsApp Chat Analyser')
st.sidebar.title('WhatsApp Chat Analyser')

users_file = st.file_uploader('Choose a file')

def create_dataframe():
    # Initialize df as None
    df = None

    if users_file is not None:
        # Convert the UploadedFile object to bytes or string
        content = users_file.read().decode('utf-8')

        # Process the content of the file
        def process_chat_line(line):
            match = re.match(r'(\d+/\d+/\d+, \d+:\d+\s?[APM]{2}) - (.*?): (.*)', line)
            if match:
                date_time, sender, message = match.groups()
                return date_time, sender, message
            else:
                return None, None, None
            
        # Process the chat data
        chat_data = content.split('\n')
        processed_data = [process_chat_line(line) for line in chat_data]

        # Filter out None entries
        processed_data = [entry for entry in processed_data if entry[0] is not None]

        # Convert processed data to DataFrame and assign column names
        df = pd.DataFrame(processed_data, columns=["DateTime", "Sender", "Message"])
    
    return df

df = create_dataframe()

if df is not None:
    # Processing the Date Column
    def preprocessing(df):
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['Day'] = df['DateTime'].dt.day
        df['AM/PM'] = df['DateTime'].dt.strftime('%p')
        df['Hour'] = df['DateTime'].dt.hour

        df['Hour_12'] = df['Hour'] % 12
        df['Hour_12'] = df['Hour_12'].replace(0, 12)

        return df

    # Display the DataFrame
    new_df = preprocessing(df)
    # st.write('DataFrame after preprocessing')
    # print(new_df.columns)

    stop_words = set(stopwords.words('english'))
    # Predefined emoji pattern
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"  # emoticons
                            "\U0001F300-\U0001F5FF"  # symbols & pictographs
                            "\U0001F680-\U0001F6FF"  # transport & map symbols
                            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "\U00002702-\U000027B0"  # additional symbols
                            "\U000024C2-\U0001F251" 
                            "]+", flags=re.UNICODE)
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    def clean_and_process_text(text):
        # Lowercase the text
        text = text.lower()
        
        # Split the text into segments that are either emojis or non-emojis
        segments = re.split(f'({emoji_pattern.pattern})', text)
        
        # Process non-emoji segments
        cleaned_segments = []
        # link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        for segment in segments:
            if re.match(emoji_pattern, segment):
                 # If the segment is an emoji, keep it as is
                cleaned_segments.append(segment)
            elif re.match(link_pattern, segment):
                cleaned_segments.append(segment)
            else:
                # Otherwise, clean the text segment
                segment = re.sub(r'[^\w\s]', '', segment)
                words = segment.split()
                filtered_words = [word for word in words if word not in stop_words]
                lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
                cleaned_segments.append(' '.join(lemmatized_words))
                
        # Join the segments back together
        processed_text = ''.join(cleaned_segments)
            
        return processed_text
    
    df['Message'] = df['Message'].apply(clean_and_process_text)

    options = np.append(df['Sender'].unique(), 'Overall')
    users_selection = st.sidebar.selectbox('Choose an option from the given options?', options)

    words = []
    for message in df['Message']:
        words.extend(message.split())

    def count_links(df, user=None):
        link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        if user is None:
            links = [link_pattern.findall(message) for message in df['Message']]
            total_links = sum(len(link_list) for link_list in links)
        elif user is not None:
            user_messages = df[df['Sender'] == user]['Message']
            links = [link_pattern.findall(message) for message in user_messages]
            total_links = sum(len(link_list) for link_list in links)
            
        return total_links

    messages = []
    for message in df['Message']:
        messages.append(message)

    total_links = count_links(df)

    def find_top_15_words(combined_text):
        words = combined_text.split()
        word_counts = Counter(words)
        top_15_words = word_counts.most_common(15)
        
        return top_15_words

    combined_text = ' '.join(df['Message'].tolist())
    top_15_words = find_top_15_words(combined_text)
    words_df = pd.DataFrame(top_15_words, columns=['Words', 'Occurrences'])

    def activity_heatmap(df):
        heatmap_data = df.pivot_table(index=df['DateTime'].dt.day_name(), columns='Hour_12', values='Message', aggfunc='count', fill_value=0)
        return heatmap_data
    
    def extract_emojis(text):
        return ''.join(c for c in text if c in emoji.EMOJI_DATA)

    def most_common_emojis(df, user=None):
        if user is None:
            all_emojis = ''.join(df['Message'].apply(extract_emojis))
            emoji_counts = Counter(all_emojis).most_common(10)
        elif user is not None:
            all_emojis = ''.join(df[df['Sender'] == user]['Message'].apply(extract_emojis))
            emoji_counts = Counter(all_emojis).most_common(10)
        return pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])

    def no_of_emoji_send_by_participant(df):
        emoji_counter = Counter()
        
        for index, row in df.iterrows():
            text = row['Message']
            sender = row['Sender']
            
            emojis = re.findall(emoji_pattern, text)
            emoji_counter[sender] += len(emojis)
            
        return emoji_counter
    
    emoji_counts = no_of_emoji_send_by_participant(df)

    # Convert the Counter to a DataFrame for better visualization
    emoji_counts_df = pd.DataFrame(emoji_counts.items(), columns=['Sender', 'Emoji Count'])
    emoji_counts_df = emoji_counts_df.sort_values(by='Emoji Count', ascending=False)

    # Emotions Analysis using VADER
    def analyze_sentiment_vader(message):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(message)
        # Compound score: -1 to 1 (negative to positive)
        return scores['compound']
    
    df['Sentiment'] = df['Message'].apply(analyze_sentiment_vader)

    if users_selection == 'Overall':
        st.title('WhatsApp Chat Analyser (Overall)')
        col1, col2= st.columns(2)

        with col1:
            st.header('Total Messages')
            st.title(len(df['Message'].unique()))

            st.header('Total Participants')
            st.title(len(df['Sender'].unique()))
        with col2:
            st.header('Total Words')
            st.title(len(words))

            st.header('Total Links')
            st.title(total_links)

        st.header('Most Common Words')
        text = ' '.join(df['Message'].dropna().astype(str))

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud using matplotlib and Streamlit
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To suppress deprecation warnings
        plt.figure(figsize=(8, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

        # Top 15 Words
        st.header('Top 15 Words')
        st.table(words_df)
        
        # No. of Messages done by each participant
        st.header('No. of Messages done by each participant')
        st.bar_chart(df['Sender'].value_counts())

        # Most Active Years
        st.header('Most Active Years')
        st.bar_chart(df['Year'].value_counts())

        # Most Active Months
        st.header('Most Active Months')
        st.bar_chart(df['Month'].value_counts())

        # No. of Messages done in AM/PM
        st.header('No. of Messages done in AM/PM')
        st.bar_chart(df['AM/PM'].value_counts())
        
        # Most Active Hours
        st.header('Most Active Hours (12-Hour Format)')
        st.bar_chart(df['Hour_12'].value_counts())

        # Activity Heatmap
        st.header('Activity Heatmap')
        heatmap_data = activity_heatmap(df)
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='Blues', annot=True)
        st.pyplot()
        
        # Most Common Emoji
        st.header('Most Common Emojis')
        emoji_df = most_common_emojis(df)
        st.table(emoji_df)

        # No. of Emojis send by each participants
        st.header('No. of Emojis send by each participants')
        st.table(emoji_counts_df)

        # Sentiment over time
        st.header('Sentiment Over Time')
        df['Date'] = df['DateTime'].dt.date
        daily_sentiment = df.groupby('Date')['Sentiment'].mean()
        st.line_chart(daily_sentiment)


    elif users_selection != 'Overall':
        st.title('WhatsApp Chat Analyser (Participant)')

        words_for_user = []
        for message in df[df['Sender'] == users_selection]['Message']:
            words_for_user.extend(message.split())

        total_links = count_links(df, users_selection)

        lengths = []
        for message in df[df['Sender'] == users_selection]['Message']:
            lengths.append(len(message))

        col1, col2= st.columns(2)

        with col1:
            st.header('Total Messages')
            st.title(len(df[df['Sender'] == users_selection]))

            st.header('Total Links')
            st.title(total_links)
        with col2:
            st.header('Total Words')
            st.title(len(words_for_user))

            st.header('Message Length (Avg.)')
            st.title(math.floor(np.mean(lengths)))

        
        st.header('Most Common Words')
        text = ' '.join(df[df['Sender'] == users_selection]['Message'].dropna().astype(str))

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud using matplotlib and Streamlit
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To suppress deprecation warnings
        plt.figure(figsize=(8, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

        combined_text_user = ' '.join(df[df['Sender'] == users_selection]['Message'].tolist())
        top_15_words_user = find_top_15_words(combined_text_user)
        words_df_user = pd.DataFrame(top_15_words_user, columns=['Words', 'Occurrences'])

        # Top 15 Words of a user
        st.header('Top 15 Words')
        st.table(words_df_user)

        # Most Active Years of a user
        st.header('Most Active Years')
        st.bar_chart(df[df['Sender'] == users_selection]['Year'].value_counts())

        # Most Active Months of a user
        st.header('Most Active Months')
        st.bar_chart(df[df['Sender'] == users_selection]['Month'].value_counts())

        # No. of Messages done in AM/PM by a user
        st.header('No. of Messages done in AM/PM')
        st.bar_chart(df[df['Sender'] == users_selection]['AM/PM'].value_counts())
        
        # Most Active Hours of a user
        st.header('Most Active Hours (12-Hour Format)')
        st.bar_chart(df[df['Sender'] == users_selection]['Hour_12'].value_counts())

        # Most Common Emoji by a user
        st.header('Most Common Emojis')
        emoji_df = most_common_emojis(df, users_selection)
        st.table(emoji_df)

        # Sentiment over time by a user
        st.header('Sentiment Over Time')
        df['Date'] = df['DateTime'].dt.date
        daily_sentiment = df[df['Sender'] == users_selection].groupby('Date')['Sentiment'].mean()
        st.line_chart(daily_sentiment)
