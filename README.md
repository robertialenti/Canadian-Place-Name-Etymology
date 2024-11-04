# Canadian Place Name Etymology
The goal of this project is to plot the etymology of Canadian place names. I identify the etymology of Canadian places using a combination of official government sources, past research, and large language models.

## Data
I downloaded the entirety of the Canadian Geographical Names Database, which includes the name, location, and __ for all places in Canada. I selected only populated places - removing forests, mountains, and other natural features - and retained only those with a scale relevance of at least ___.

## Code

The code is divided into 6 sections.

### 1. Preliminaries
I begin by importing the libraries I need for ___.

### 2. Gathering Data

### 3. Extract Etymologies from Descriptions
Using OpenAI's API, I use a large language model to read each place name's description and classify it as either: Indigenous, English, Scottish, Welsh, Irish, French, Other European, Other, or Unknown. In the event that ____.

### 4. Merging Etymologies with Place Names

### 5. Predicting Remaining Etymologies

### 6. Plotting
Finally, I plot the place names on an interacive map using Folium. This allows users to explore the results.

## Results
Here is a static version of the map. As expected, I find that place names located in the country's Northern regions are disproportionately indigenous.

<img src="https://github.com/robertialenti/Canadian-Place-Name-Etymology/raw/main/output/etymology_map.png">
