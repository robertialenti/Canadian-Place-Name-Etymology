# Mapping Etymology of Canadian Place Names
This project visualizes the etymology of Canadian place names. I identify the origin of Canadian cities, towns, and other inhabited places using a combination of official government sources, past research, and large language models. This project was inspired by Jay Foreman's "[Why are British place names so hard to pronounce?](https://www.youtube.com/watch?app=desktop&v=uYNzqgU7na4)", which maps the origin of place names in the United Kingdom. As far as I know, the spatial distribution of place names' etymolgogy in Canada has yet to be publicly documented at the time this repository was published. 

## Data
I downloaded the entirety of the Canadian Geographical Names Database (CGND), which includes the name, description, and geolocation for more than 360,000 places in Canada. For indigenous place names, the CGND also provides the language of origin, which allows me to easily identify places with indigenous origins. More details can be found [here](https://natural-resources.canada.ca/earth-sciences/geography/geographical-names-board-canada/about-canadian-geographical-names-database/9180).

## Code
The code is divided into 6 sections. To run the code without modification, create a project directory with `code`, `data`, and `output` folders. A valid OpenAI API key must be passed into the appropriate argument to run 3. Extract Etymology from Descriptions.

### 1. Preliminaries
I begin by importing the libraries I need for manipulating data, parsing HTML, using large language models for classification, and mapping results.

### 2. Gathering Data
In the CGND, I select only populated places - removing forests, mountains, and other natural features. I retain only places in the CGND with a scale relevance - indicating whether or not a place name should be displayed given the scale of a map - of at least 250,000. Practically, this removes the smallest and least significant populated places in the country. For any place name with a value of Language that is not "Undetermined", the etymology is classified as indigenous. I remove duplicates in terms of place name-province name.

Next, I gather etymologyical information for 1,118 populated places from The Origin and Meaning of Place Names in Canada (Armstrong, 1930), which remains a referential text on the topic. For each place name discussed, Armstrong provides some historical, geographical, and etymological information. For example, here is the excerpt for Montreal, QC:

_Montreal: the largest and in many respects the most important city in Canada, situated on the island of Montreal in the province of Quebec. The site of the city was occupied originally by the Algonquin village of Hochelaga. In 1642, the French founded the present city under the name of Ville-Marie, de Maisonneuve being in command of the party of colonists. A prominent elevation near the city had been named Mont Réal (“Mount Royal”) by the French. Since the English captured the town in 1760 it has been known as Montreal, (Mont Réal). Montreal River, named after the metropolis, in Algoma, Ontario, emptying into Lake Superior, possesses much potential electric power._

### 3. Extract Etymologies from Descriptions
Using OpenAI's API, I use the GPT-4 large language model to read each place name's description and subsequently classify its origin as either: Indigenous, English, Scottish, Welsh, Irish, French, Other European, Other, or Unknown. With manual validation, I ensure that the model is able to correctly classify the majority of place names in Armstorng (1930). Montreal, whose description is shown above, is correctly classified as having an indigenous origin.

### 4. Merging Etymologies with Place Names
I merge the now-classified place names from Armstrong (1930) to the CGND using a combination of exact and fuzzy merging on place name and province. I am able to match all of the place names discussed in Armstrong (1930). The merged data, with approximately 1,400 place names, is assumed to have the correct etymology.

### 5. Predicting Remaining Etymologies
I predict the etymology of the remaining place names using a pre-trained large language model - in particular, a distiled version of the RoBERTa base model - hosted on Hugging Face. I fine-tune the model using the set of 1,400 place names for which I already have etymological informaiton. I validate the model's performance, ensuring that the loss derived from generating predictions on a subset of the data is below ____.

I manually validate that ____.

### 6. Plotting
Finally, I plot the place names on an interacive map using Folium. This allows users to explore the results in greater detail.

## Results
Here is a tabulation of place name origin, by province or territory.

| Province | Indigenous | English | Scottish | Welsh | Irish | French | Other European | Other | Total |
| -------- | ---------- | ------- | -------- | ----- | ----- | ------ | -------------- | ----- | ----- |
| British Columbia | 436 | 962 | 128 | 0 | 1 | 52 | 18 | 0 | 1 | 1598 |
| Alberta | 314 | 865 | 150 | 0 | 0 | 71 | 30 | 0 | 0 | 1430 |
| Saskatchewan | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Manitoba | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Ontario | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Quebec | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Newfoundland and Labrador | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| New Brunswick | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Nova Scotia | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Prince Edward Island | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Yukon | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Northwest Territories | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Nunavut | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Total | 3833 | 11266 | 1385 | 9 | 52 | 2445 | 241 | 2 | 6 | 19229 |

Here is a static version of the map. As expected, I find that place names located in the country's Northern regions are more likely to have indigenous origins. Central and Western Canada have place names that are mostly of British, Irish, or other European origin. The prominence of place names with French origin increases dramatically when crossing from Ontario into Quebec, though names of British origin can be found in the province's Eastern Townships region.

<img src="https://github.com/robertialenti/Canadian-Place-Name-Etymology/raw/main/output/etymology_map.png">

In the interactive version of the map, the user can clearly see the name ___, as well as the source of the place name's etymology.
