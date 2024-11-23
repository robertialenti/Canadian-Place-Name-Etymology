""" 
This scripts geolocates Canadian city names, identifies the place name's etymology using available research, with the help of large language models, and plots results.
"""

#%% 1. Preliminaries
# General
import pandas as pd
import folium
import os
import re
import io
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm
from fuzzywuzzy import fuzz, process
from unidecode import unidecode

# Mapping
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# Large Language Modelling
from openai import OpenAI
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Options
pd.set_option("display.expand_frame_repr", False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project Directory
filepath = "C:/Users/Robert/OneDrive/Desktop/Bobby/GitHub/Canadian-Place-Name-Etymology/"


#%% 2. Gathering Data
# Define Function for Gathering Place Name Data From Canadian Geographical Names Database
def gather_data_cgnd():
    df_places = pd.read_csv(filepath + "data/place_names.csv", encoding='ISO-8859-1')
    
    # Rename Variables
    df_places = df_places.rename(columns = 
                                 {"Geographical Name": "place_name",
                                  "Generic Category": "category",
                                  "Generic Term": "term",
                                  "Relevance at Scale": "relevance",
                                  "Language": "language",
                                  "Latitude": "lat",
                                  "Longitude": "long",
                                  "Province - Territory": "province"})
    
    # Retain Only Populated Places
    df_places = df_places[df_places["category"] == "Populated Place"]
    
    # Retain Only Highly Relevant Place Names
    df_places = df_places[df_places["relevance"] >= 250000]
    
    # Remove Duplicates of Place Name and Province
    df_places = df_places.drop_duplicates(subset=['place_name', 'province'], keep = "first")
    
    # Return Data
    return df_places


df_places = gather_data_cgnd()


# Define Function for Gathering Etymological Data From Armstrong (1930)
def gather_data_armstrong():
    file = filepath + "data/etymology.html"
    
    # Read HTML File
    def read_html(file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    # Get soup from the HTML file
    soup = read_html(file)
    
    # Check if soup is None
    if soup is None:
        print("Error: Failed to parse HTML file.")
        return pd.DataFrame()  # Return an empty DataFrame if reading failed
    
    # Initialize Lists to Hold Place Names and Descriptions
    place_names = []
    descriptions = []
    
    # Extract Place Names and Descriptions
    current_description = ""
    last_place_name = ""
    for p in soup.find_all('p'):
        span = p.find('span', class_='bold')
        
        if span:
            if current_description:
                place_names.append(last_place_name)
                descriptions.append(current_description.strip())
                current_description = ""
    
            last_place_name = span.text.strip()
        current_description += p.text.strip() + " "
    
    # Check if there's a remaining description to store after the loop
    if current_description:
        place_names.append(last_place_name)
        descriptions.append(current_description.strip())
    
    # Create Dataframe with Place Names and Descriptions
    df_etymology = pd.DataFrame(
        {'place_name': place_names,
         'description': descriptions})
    
    # Extract Province from Description
    provinces = ["British Columbia", "Alberta", "Saskatchewan", "Manitoba", "Ontario", "Quebec", 
                 "New Brunswick", "Nova Scotia", "Prince Edward Island", "Newfoundland and Labrador", 
                 "Nunavut", "Northwest Territories", "Yukon"]
    
    def classify_province(description, provinces):
        # Initialize a dictionary to store the first occurrence of each province
        province_positions = {}
        
        # Iterate through each province and find its position in the description
        for province in provinces:
            # Use a case-insensitive search for province in the description
            match = re.search(re.escape(province), description, re.IGNORECASE)
            if match:
                province_positions[province] = match.start()
        
        # If no provinces were found, return "Unknown"
        if not province_positions:
            return "Unknown"
        
        # Return the province that appears first in the description
        return min(province_positions, key=province_positions.get)
    
    # Apply the function to each row in the DataFrame
    df_etymology["province"] = df_etymology["description"].apply(lambda x: classify_province(x, provinces))
    
    return df_etymology

# Define the filepath correctly before calling the function
df_etymology = gather_data_armstrong()


#%% 3. Extract Etymology from Descriptions
# Define Function for Summarizing Etymological Descriptions for Place Names in Armstrong (1930)
def summarize_descriptions(place_name, description):
    # Define Valid Etymologies
    valid_etymologies = ["Indigenous", "English", "Scottish", "Irish", "Welsh", "French", "Other European", "Other", "Unknown"]
    
    # Initialize Client
    client = OpenAI(api_key = 'api_key')
    
    # Generate Response
    response = client.chat.completions.create(
        model = "gpt-4",
        temperature = 0.20,
        messages=[
            {"role": "system", "content": "You are an expert in Canadian place names etymologies."},
            {"role": "user", "content": f"Given the following Canadian place name and its etymological history, strictly classify the place name as one of the following: 'Indigenous', 'English', 'Scottish', 'Irish', 'Welsh', 'French', 'Other European', 'Other', or 'Unknown': \n\n Place Name: {place_name}, Descrption: {description}"}
        ]
    )
    
    # Select First Response
    etymology = response.choices[0].message.content.strip().replace("'", "")

    # Attempt to Extract Valid Etymology from Response
    if etymology not in valid_etymologies:
        for valid_etymology in valid_etymologies:
            if valid_etymology in etymology:
                etymology = valid_etymology
                return etymology
    
    # Reattempt Classification if Initial Classification is Invalid
    if etymology not in valid_etymologies:
        print(f"Etymology for {place_name} is not a valid classification. Trying again.")
        
        # Regenerate Response
        response = client.chat.completions.create(
            model = "gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in Canadian place names etymologies."},
                {"role": "user", "content": f"Try again. You are summarizing the etymological history, rather than classifying. Please ensure that the place name is strictly classified as one of the following: 'Indigenous', 'English', 'Scottish', 'Irish', 'Welsh', 'French', 'German', 'Scandinavian', 'Other European', 'Other', or 'Unknown': \n\n Place Name: {place_name}, Description: {description}"}
            ]
        )
        
        # Select First Response
        etymology = response.choices[0].message.content.strip()
        
        # Classify Etymology as Unknown
        if etymology not in valid_etymologies:
            print(f"Unable to confidently identify the etymology for {place_name}. Classifying as unknown etymology.")
            etymology = "Unknown"
        
    # Return Etymology for Place Name
    return etymology


# Classify Each Place Name's Etymology
for index in tqdm(df_etymology.index):
    place_name = df_etymology.loc[index, "place_name"]
    description = df_etymology.loc[index, "description"]
    etymology = summarize_descriptions(place_name, description)
    df_etymology.loc[index, "etymology"] = etymology
    print(f"{df_etymology.loc[index, 'place_name']} classified as: {etymology}")
    df_etymology.to_csv(filepath + "data/categorized_place_names.csv")
    
    
#%% 4. Merge Etymologies with Place Names
df_etymology = pd.read_csv(filepath + "data/categorized_place_names.csv")

# Define Function for Merging Etymologies with Place Names
def merge(df_places, df_etymology, threshold):
    # Clean Name Places and Provinces
    for var in ["place_name", "province"]:
        for df in [df_places, df_etymology]:
            df[var] = df[var].apply(unidecode)
            df[var] = df[var].str.strip()
    
    # Attempt Exact Match Merge
    df_places2 = pd.merge(df_places,
                          df_etymology,
                          on = ["place_name", "province"],
                          how = "left")
    df_places2["merge_type"] = "exact"
    
    # Classify Indigenous Place Names Using Geographical Names Database
    df_places2.loc[df_places2['language'] != 'Undetermined', 'etymology'] = 'Indigenous'
    
    # Note Source
    df_places2['source'] = 'Canadian Geographic Names Database'
    df_places2.loc[df_places2['language'] == "Undetermined", 'source'] = 'Armstrong (1930)'
    
    # Attempt Fuzzy Match Merge
    unmatched_places = df_places2[df_places2['etymology'].isna()]
    fuzzy_matches = []
    
    for i, row in tqdm(unmatched_places.iterrows()):
        place_name = row['place_name']
        province = row['province']
        
        # Filter etymology data to the same province first for better accuracy
        etymology_subset = df_etymology[df_etymology['province'] == province]
        
        if not etymology_subset.empty:
            match = process.extractOne(place_name, 
                                       etymology_subset['place_name'], 
                                       scorer = fuzz.ratio)
            
            # Verify if Match is Found
            if match:
                best_match, score = match[0], match[1]
                
                # Ensure Match Score Exceeds Threshold
                if score >= threshold:  
                    # Get the corresponding etymology data for the best match
                    matched_row = etymology_subset[etymology_subset['place_name'] == best_match]
                    fuzzy_matches.append({
                        'index': i,
                        'place_name': row['place_name'],
                        'province': row['province'],
                        'best_match': best_match,
                        'etymology': matched_row['etymology'].values[0]
                    })

    # Append Exact Matches and Fuzzy Matches
    for match in fuzzy_matches:
        df_places2.loc[match['index'], 'place_name'] = match['best_match']
        df_places2.loc[match['index'], 'etymology'] = match['etymology']
        df_places2.loc[match["index"], "merge_type"] = "fuzzy"
    
    # Return Merged DataFrame
    return df_places2

df_places2 = merge(df_places, df_etymology, threshold = 80)
df_places2 = df_places2[~df_places2["etymology"].isna()]
df_places2.to_csv(filepath + "data/df_places2.csv")


#%% 5. Predicting Place Name Etymologies
df_places2 = pd.read_csv(filepath + "data/df_places2.csv")

# Define the Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


# Define a custom compute_metrics function for accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


# Define Function for Fine-Tuning NLP Model with Accuracy
def fine_tune_with_accuracy(df_places2):
    # Create Training and Testing Sets
    x_train, x_test, y_train, y_test = train_test_split(
        df_places2["place_name"],
        df_places2["etymology"],
        test_size=0.2,
        shuffle=True
    )
    
    # Encode Etymologies
    all_classes = ['Indigenous', 'English', 'Scottish', 'Welsh', 'Irish', 'French', 'Other European', 'Other', 'Unknown']  # Use actual class names
    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)  # Fit on all classes you expect to see
    
    # Encode the labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Select Pre-Trained Model
    model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', 
                                                             num_labels=len(label_encoder.classes_))
    
    # Tokenize Data
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    def tokenize(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=32)
    
    train_encodings = tokenize(x_train.tolist())
    test_encodings = tokenize(x_test.tolist())
    
    # Convert Training and Testing Sets to PyTorch Tensors
    train_input_ids = torch.tensor(train_encodings['input_ids'])
    train_attention_mask = torch.tensor(train_encodings['attention_mask'])
    train_labels = torch.tensor(y_train_encoded, dtype=torch.long) 
    
    test_input_ids = torch.tensor(test_encodings['input_ids'])
    test_attention_mask = torch.tensor(test_encodings['attention_mask'])
    test_labels = torch.tensor(y_test_encoded, dtype=torch.long)
    
    # Prepare Training and Testing Sets
    train_dataset = TextDataset(train_input_ids, train_attention_mask, train_labels)
    test_dataset = TextDataset(test_input_ids, test_attention_mask, test_labels)
    
    # Specify Training Parameters
    training_args = TrainingArguments(
        output_dir='./results',          
        evaluation_strategy="epoch",     
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=8,    
        gradient_accumulation_steps=2,   
        num_train_epochs=3,              
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,                
        use_cpu=True
    )
    
    # Create Trainer with Custom Metrics
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=test_dataset,           
        compute_metrics=compute_metrics
    )
    
    # Fine-Tune Model
    trainer.train()
    
    # Examine Evaluation Results
    evaluation_results = trainer.evaluate()
    print(evaluation_results)
    
    # Return Model
    return trainer, label_encoder


# Call the function to fine-tune and display accuracy
trainer, label_encoder = fine_tune_with_accuracy(df_places2)

# Select Missing Place Names to Predict
df_places_missing = df_places.merge(df_places2[['place_name', 'province', 'lat', 'long', 'source']], 
                                    on=['place_name', 'province', 'lat', 'long'], 
                                    how='left', 
                                    indicator=True)
df_places_missing = df_places_missing[df_places_missing['_merge'] == 'left_only'].drop(columns=['_merge'])

# Define Function for Predicting Place Name Etymologies
def predict(df_places_missing):
    # Tokenize Data
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    def tokenize(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=32)
    
    # Tokenize the place names in df_places_missing
    missing_encodings = tokenize(df_places_missing["place_name"].tolist())
    
    # Convert tokenized inputs to PyTorch tensors
    missing_input_ids = torch.tensor(missing_encodings['input_ids'])
    missing_attention_mask = torch.tensor(missing_encodings['attention_mask'])
    
    # Create a Dataset for df_places_missing
    missing_dataset = TextDataset(missing_input_ids, missing_attention_mask, labels=None)  # No labels since it's for prediction
    
    # Predictions on the missing dataset
    missing_predictions = trainer.predict(missing_dataset)
    
    # Get predicted labels
    missing_predicted_labels = torch.argmax(torch.tensor(missing_predictions.predictions), dim=-1)
    
    # Decode the predicted labels
    missing_predicted_labels_decoded = label_encoder.inverse_transform(missing_predicted_labels.cpu().numpy())
    
    # Add the predictions to the DataFrame
    df_places_missing['predicted_etymology'] = missing_predicted_labels_decoded
    df_places_missing["source"] = "Predicted"
    
    # Return Predictions
    return df_places_missing


df_places_missing = predict(df_places_missing)

df_places3 = pd.concat([df_places2, df_places_missing[["place_name", "province", "predicted_etymology", "source", "lat", "long"]]]).reset_index()
df_places3.loc[df_places3['source'] == 'Predicted', 'etymology'] = df_places3['predicted_etymology']
df_places3.to_csv(filepath + "data/df_places3.csv")


#%% 6. Plotting
df_places3 = pd.read_csv(filepath + "data/df_places3.csv")

# Tabulation
etymology_counts = df_places3.groupby(['province', 'etymology']).size().unstack(fill_value=0)
etymology_counts['Total'] = etymology_counts.sum(axis=1)

# Sort columns in the specified order, adding any missing ones with 0 values
desired_order = ['Indigenous', 'English', 'Scottish', 'Welsh', 'Irish', 'French', 'Other European', 'Other', 'Unknown', 'Total']
for col in desired_order:
    if col not in etymology_counts.columns:
        etymology_counts[col] = 0

etymology_counts = etymology_counts[desired_order]

# Add a row for totals across all provinces
etymology_counts.loc['Total'] = etymology_counts.sum()

# Define Function for Plotting Place Name Etymologies on Map
def plot_etymologies(data):  
    # Center Map
    m = folium.Map(location=[56.1304, -106.3468], 
                   zoom_start = 4)
    
    # Launch Application
    app = QApplication([])
    
    # Create Markers for Place Names
    for i, row in data.iterrows():
        popup_text = f"Place Name: {row['place_name']}<br>Etymology: {row['etymology']}<br>Source: {row['source']}"
        etymology_colors = {
        'Indigenous': 'green',
        'English': 'red',
        'Scottish': 'crimson',
        'Welsh': 'maroon',
        'Irish': 'orange',
        'French': 'blue',
        "Other European": "purple",
        'Other': 'black'
    }
        marker_color = etymology_colors.get(row['etymology'], 'gray')
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_text, max_width = 300),
            fill = True,
            radius = 1,
            opacity = 0.7,
            fill_opacity = 0.7,
            color = marker_color,
            fill_color = marker_color
        ).add_to(m)
    
    # Configure Legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 200px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:6px; padding:10px;">
        <b>Legend</b><br>
        <div>
            <div style="background-color: green; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Indigenous<br>
        </div>
        <div>
            <div style="background-color: red; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;English<br>
        </div>
        <div>
            <div style="background-color: crimson; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Scottish<br>
        </div>
        <div>
            <div style="background-color: maroon; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Welsh<br>
        </div>
        <div>
            <div style="background-color: orange; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Irish<br>
        </div>
        <div>
            <div style="background-color: blue; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;French<br>
        </div>
        <div>
            <div style="background-color: purple; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Other European<br>
        </div>
        <div>
            <div style="background-color: black; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></div>
            &nbsp;Other<br>
        </div>
    </div>
    '''
    
    # Add Legend to Map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the Map as HTML and PNG File
    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save(filepath + "output/etymology_map.png")
    
    # Launch the Application
    view = QWebEngineView()
    view.load(QUrl.fromLocalFile(filepath + "output/etymology_map.html"))
    view.show()
    app.exec_()


plot_etymologies(df_places3)
