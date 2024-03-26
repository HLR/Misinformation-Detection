annotation_level_0 = {
    'no annotation': '0',
    'has annotation': '1'
}

annotation_level_1 = {
  "narrative with details": "1",
  "using anecdotes and personal experience as evidence": "2",
  "distrusting government or corporations": "3",
  "politicizing health issues": "4",
  "highlighting uncertainty and risk": "5",
  "exploiting science’s limitations": "6",
  "inappropriate use of scientific and other evidence": "7",
  "rhetorical tricks": "8",
  "biased reasoning to make a conclusion": "9",
  "emotional appeals": "10",
  "distinctive linguistic features": "11",
  "establishing legitimacy": "12",
}

annotation_level_2 = {
  "narrative with details_verified to be false": "1a",
  "narrative with details_details verified to be true": "1b",
  "narrative with details_details not verified": "1c",
  "financial motivation": "3a",
  "freedom of choice and agency": "4a",
  "ingroup vs. outgroup": "4b",
  "political figures or political argument": "4c",
  "religion or ideology": "4d",
  "inappropriate use of scientific and other evidence - out of context_verified": "7a",
  "less robust evidence or outdated evidence_verified": "7b",
  "exaggeration/absolute language": "8a",
  "selectively omission": "8b",
  "inappropriate analogy or false connection": "9a",
  "wrong cause-effect": "9b",
  "claims without evidence": "9c",
  "evidence does not support conclusion": "9d",
  "shifting hypothesis": "9e",
  "fear": "10a",
  "anger": "10b",
  "hope": "10c",
  "anxiety": "10d",
  "uppercase words": "11a",
  "linguistic intensifier (e.g., extreme words)": "11b",
  "title of article as clickbait": "11c",
  "bolded, underline or italicized": "11d",
  "ellipses, exaggerated/excessive usage of punctuation marks": "11e",
  "citing source to establish legitimacy": "12a",
  "legitimate persuasive techniques (e.g., multiple sources, broad range of appeals, metaphor, humor, rhetorical question)": "12b",
  "surface credibility markers": "12c",
  "call to action": "12d"
}

annotation_level_3 = {
  "citing source to establish legitimacy_source verified to be credible": "12ai",
  "citing source to establish legitimacy_source verified to not be credible in this context": "12aii",
  "citing source to establish legitimacy_source not verified": "12aiii",
  "citing source to establish legitimacy_source verified to be made up": "12aiv",
  "legitimate persuasive techniques: rhetorical question": "12bi",
  "legitimate persuasive techniques: humor": "12biii",
  "surface credibility markers - medical jargon": "12ci",
  "surface credibility markers - words associated with nature or healthiness": "12cii",
  "surface credibility markers - words associated with uncertainty and anxiety": "12ciii",
  "surface credibility markers - simply claiming authority or credibility": "12civ",
}

def normalize_annotations(anno):
    anno = anno.lower()
    if anno in ["distrusting government or pharmaceutical companies", "distrusting government or corporations"]:
        anno = "distrusting government or corporations"
    elif anno in ["inappropriate use of scientific evidence", "inappropriate use of scientific and other evidence"]:
        anno = "inappropriate use of scientific and other evidence"
    elif anno in ["rhetorical tricks", "rhetorical questions"]:
        anno = "rhetorical tricks"
    elif anno in ["inappropriate use of scientific and other evidence - out of context_verified", "inappropriate use of scientific evidence - out of context_verified"]:
        anno = "inappropriate use of scientific and other evidence - out of context_verified"
    elif anno in ["less robust evidence or outdated evidence_verified", "less robust evidence or outdated evidence_verify"]:
        anno = "less robust evidence or outdated evidence_verified"
    elif anno in ["exaggeration", "exaggeration/absolute language"]:
        anno = "exaggeration/absolute language"
    elif anno in ["title of article", "title of article as clickbait"]:
        anno = "title of article as clickbait"
    elif anno in ["lack of evidence or use unverified and incomplete evidence to make a claim", "lack of citation for evidence", "claims without evidence"]:
        anno = "claims without evidence"
    elif anno in ["bolded words or underline", "bolded,  underline or italicized", "bolded, underline or italicized"]:
        anno = "bolded, underline or italicized"
    elif anno in ["citing seemingly credible source", "citing seemingly credible sources to strengthen one’s arguments", "citing source to establish legitimacy"]:
        anno = "citing source to establish legitimacy"
    elif anno in ["surface credibility markers - medical or scientific jargon", "surface credibility markers - medical jargon"]:
        anno = "surface credibility markers - medical jargon"
    return anno
    
def get_low_freq():
    return ["6","11b","4a","4b","4c","4d","8b","9b","12aiv","12biii", "12ciii"]

def get_annotations():
    return {**annotation_level_1, **annotation_level_2, **annotation_level_3}

def get_annotation_keys():
    return get_annotations().values()

def get_annotation_layer(layer):
    if int(layer) == 1:
        target = annotation_level_0
    elif int(layer) == 2:
        target = annotation_level_1
    elif int(layer) == 3:
        target = annotation_level_2
    elif int(layer) == 4:
        target = annotation_level_3
    return dict(list(filter(lambda x:x[1] not in get_low_freq(),target.items())))
        
def get_annotation_layer_keys(layer):
    layer_dict = get_annotation_layer(layer)
    res = sorted(list(set(list(layer_dict.values()))))
    return res

def get_annotation_layer_names(layer):
    values = get_annotation_layer_keys(layer)
    layer_dict = get_annotation_layer(layer)
    layer_dict_keys = [x for x in layer_dict.keys() if layer_dict[x] in values]
    res = sorted(layer_dict_keys,key=lambda x:values.index(layer_dict[x]))
    return res

def get_annotation_layer_and_indexes():
    return dict(list(enumerate(annotation_level_1.keys()))), dict(list(enumerate(annotation_level_2.keys()))), dict(list(enumerate(annotation_level_3.keys())))