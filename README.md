# Mapping Etymology of Canadian Place Names
The goal of this project is to plot the etymology of Canadian place names. I identify the etymology of Canadian places using a combination of official government sources, past research, and large language models.

## Data
I downloaded the entirety of the Canadian Geographical Names Database, which includes the name, location, and __ for all places in Canada. I selected only populated places - removing forests, mountains, and other natural features - and retained only those with a scale relevance of at least ___.

## Code

The code is divided into 6 sections. To run the code without modification, create a project directory with `code`, `data`, and `output` folders. Place ____.

### 1. Preliminaries
I begin by importing the libraries I need for manipulating data, parsing HTML, using large language models for classification, and mapping results.

### 2. Gathering Data
I gather place name data from the Canadian Geographical Names Database (CGND). This publicly available database includes the names, location, ___, and ___ of more than ___ Canadian place names, including cities, towns, mountains, lakes, railroad stations, etc. I retain only populated places with a relevance of at leasts 250,000 - meaning that _____.

I gather etymologyical information for approximately 2,000 populated places from ___ (Armstrong, 1930), which provides an important ___.

### 3. Extract Etymologies from Descriptions
Using OpenAI's API, I use a large language model to read each place name's description and classify it as either: Indigenous, English, Scottish, Welsh, Irish, French, Other European, Other, or Unknown. In the event that ____.

### 4. Merging Etymologies with Place Names

I merge ___ from Armstrong (1930) to the CGND using a combination of exact and fuzzy merging on place name and province. I am able to match ____.

### 5. Predicting Remaining Etymologies

I predict the etymology of the remaining place names using a pre-trained large language model - in particular, a distiled version of the RoBERTa base model - hosted on Hugging Face. I fine-tune the model using the set of more than 2,000 place names for which I already have etymological informaiton. I validate ___, ensuring that the loss derived from generating predictions on a subset of the data is below ____.

I manually validate that ____.

### 6. Plotting
Finally, I plot the place names on an interacive map using Folium. This allows users to explore the results in greater detail.

## Results
Here is a static version of the map. As expected, I find that place names located in the country's Northern regions are more likely to have indigenous origins. Central and Western Canada have place names that are mostly of British, Irish, or other European origin. The prominence of place names with French origin increases dramatically when crossing from Ontario into Quebec, though names of British origin can be found in the province's Eastern Townships region.

<img src="https://github.com/robertialenti/Canadian-Place-Name-Etymology/raw/main/output/etymology_map.png">
