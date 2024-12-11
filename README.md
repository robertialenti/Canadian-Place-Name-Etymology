# Mapping Etymology of Canadian Place Names
This project visualizes the etymology of Canadian place names. I identify the origin of Canadian cities, towns, and other inhabited places using a combination of official government sources, past research, and large language models. This project was inspired by Jay Foreman's "[Why are British place names so hard to pronounce?](https://www.youtube.com/watch?app=desktop&v=uYNzqgU7na4)", which maps the origin of place names in the United Kingdom. As far as I know, the spatial distribution of place names' etymolgogy in Canada has yet to be publicly documented at the time this repository was published. 

## Data
I downloaded the entirety of the Canadian Geographical Names Database (CGND), which includes the name, description, and geolocation for more than 360,000 places in Canada. For indigenous place names, the CGND also provides the language of origin, which allows me to easily identify places with indigenous origins. More details can be found [here](https://natural-resources.canada.ca/earth-sciences/geography/geographical-names-board-canada/about-canadian-geographical-names-database/9180). 

## Code
The code is divided into 6 sections. To run the code without modification, create a project directory with `code`, `data`, and `output` folders. A valid OpenAI API key must be passed into the appropriate argument to run Section 3. Extract Etymology from Descriptions.

### 1. Preliminaries
I begin by importing the libraries I need for manipulating data, parsing HTML, using large language models for classification, and mapping results.

### 2. Gathering Data
In the CGND, I select only populated places - removing forests, mountains, and other natural features. I retain only places in the CGND with a scale relevance - indicating whether or not a place name should be displayed given the scale of a map - of at least 250,000. Practically, this removes the smallest and least significant populated places in the country. For any place name with a value of Language that is not "Undetermined", the etymology is classified as indigenous. I remove duplicates in terms of place name-province name.

Next, I gather etymologyical information for 1,118 populated places from The Origin and Meaning of Place Names in Canada (Armstrong, 1930), which remains a referential text on the topic. For each place name discussed, Armstrong provides some historical, geographical, and etymological information. For example, here is the excerpt for Montreal, QC:

_Montreal: the largest and in many respects the most important city in Canada, situated on the island of Montreal in the province of Quebec. The site of the city was occupied originally by the Algonquin village of Hochelaga. In 1642, the French founded the present city under the name of Ville-Marie, de Maisonneuve being in command of the party of colonists. A prominent elevation near the city had been named Mont Réal (“Mount Royal”) by the French. Since the English captured the town in 1760 it has been known as Montreal, (Mont Réal). Montreal River, named after the metropolis, in Algoma, Ontario, emptying into Lake Superior, possesses much potential electric power._

### 3. Extract Etymologies from Descriptions
Using OpenAI's API, I use the GPT-4 large language model to read each place name's description and subsequently classify its origin as either: Indigenous, English, Scottish, Welsh, Irish, French, Other European, Other, or Unknown. With manual validation, I ensure that the model is able to correctly classify the majority of place names in Armstrong (1930). Montreal, whose description is shown above, is correctly classified as having an indigenous origin.

### 4. Merging Etymologies with Place Names
I merge the now-classified place names from Armstrong (1930) to the CGND using a combination of exact and fuzzy merging on place name and province. I am able to match all of the place names discussed in Armstrong (1930). I assume that the resulting merged dataset, with approximately 1,400 place names, has correctly classified each place name's etymology.

### 5. Predicting Remaining Etymologies
I predict the etymology of the remaining place names using a pre-trained large language model - in particular, a distilled version of the RoBERTa base model - hosted on Hugging Face. I fine-tune the model using the set of 1,400 place names for which I already have etymological information. The model's performance is evaluated using te F score, which balances precision and recall. This is done to ensure that the model's performance is not inflated by simply predicting the most common classes. The confusion matrix describes the relationship between true positives and false positive predictions in the testing set. Naturally, I accept that some of the place names will have their etymologies incorrectly classified, as one can confirm using manual validation.

### 6. Plotting
Finally, I plot the place names on an interacive map using Folium. This allows users to explore the results in greater detail. Each place name is colored by its etymology, when hovering over the place name, the source of the classification is provided.

## Results
Here is a tabulation of place name origin, by province or territory.

| Province | Indigenous | English | Scottish | Welsh | Irish | French | Other European | Other | Unknown | Total |
| -------- | ---------- | ------- | -------- | ----- | ----- | ------ | -------------- | ----- | ------- | ----- |
| British Columbia | 405 | 939 | 116 | 0 | 3 | 127 | 7 | 0 | 1 | 1598 |
| Alberta | 285 | 837 | 145 | 0 | 4 | 144 | 15 | 0 | 0 | 1430 |
| Saskatchewan | 315 | 972 | 194 | 0 | 7 | 179 | 21 | 0 | 0 | 1688 |
| Manitoba | 205 | 611 | 105 | 0 | 2 | 132 | 8 | 0 | 0 | 1063 |
| Ontario | 695 | 3291 | 481 | 9 | 40 | 381 | 50 | 2 | 4 | 4953 |
| Quebec | 489 | 646 | 93 | 0 | 5 | 1709 | 4 | 0 | 1 | 2947 |
| Newfoundland and Labrador | 118 | 665 | 28 | 0 | 1 | 119 | 1 | 0 | 0 | 932 |
| New Brunswick | 272 | 1145 | 74 | 0 | 3 | 330 | 6 | 0 | 0 | 1830 |
| Nova Scotia | 286 | 1439 | 75 | 0 | 2 | 192 | 3 | 0 | 0 | 1997 |
| Prince Edward Island | 55 | 389 | 35 | 0 | 4 | 54 | 1 | 0 | 0 | 538 |
| Yukon | 37 | 67 | 8 | 0 | 0 | 12 | 1 | 0 | 0 | 125 |
| Northwest Territories | 29 | 36 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 70 |
| Nunavut | 29 | 23 | 2 | 0 | 0 | 4 | 0 | 0 | 0 | 58 |
| Total | 3219 | 11061 | 1358 | 9 | 71 | 3386 | 117 | 2 | 6 | 19229 |

Here is a static version of the map. As expected, I find that place names located in the country's Northern regions are more likely to have indigenous origins. Central and Western Canada have place names that are mostly of British, Irish, or other European origin. The prominence of place names with French origin increases dramatically when crossing from Ontario into Quebec, though names of British origin can be found in the province's Eastern Townships region.

<img src="https://github.com/robertialenti/Canadian-Place-Name-Etymology/raw/main/output/etymology_map.png">

You can also view an [interactive map](https://robertialenti.github.io/Canadian-Place-Name-Etymology/output/etymology_map.html).

