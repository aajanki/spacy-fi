import re
import json
from itertools import permutations

class SpamClassifier:
    def __init__(self, modelfile):
        with open(modelfile) as f:
            self.model_parameters = json.load(f)

    def extract_features(self, text):
        words = (x.group(0) for x in re.finditer(r'\w[-\w]*\w', text.lower()))
        return {word: 1 for word in words}

    def score(self, features):
        score = self.model_parameters['bias'] + self.model_parameters.get('threshold', 0)
        weights = self.model_parameters['weights']
        for feat_name, feat_value in features.items():
            score += feat_value * weights.get(feat_name, 0)

        return score

    def predict(self, text):
        return self.score(self.extract_features(text)) < 0


unigrams = '( ) [ ] { } < > = - + " \' ; : ! & \\ | _'.split()
bigrams = list(x[0] + x[1] for x in permutations(unigrams, 2))
trigrams = list(x[0] + x[1] + x[2] for x in permutations(unigrams, 3))
feature_names = unigrams + bigrams + trigrams

class CodeClassifier:
    def __init__(self, modelfile):
        with open(modelfile) as f:
            self.model_parameters = json.load(f)

    def extract_features(self, text):
        features = {}

        # word count
        plain_words = re.finditer(r'\b\w\w+\b', text)
        word_count = max(sum(1 for _ in plain_words), 1)
        features['word_count'] = word_count

        # punctuation features
        text_no_space = re.sub(r'\s+', '', text)
        for feat in feature_names:
            c = text_no_space.count(feat)
            crel = round(c / len(text) * 1000)
            if c > 0:
                features[feat] = c
            if crel > 0:
                features['relative ' + feat] = crel

        # html tags
        html_feature_regexs = {
            '<script>': r'<script[ >]|</script>',
            '<iframe>': r'<iframe[ >]|</frame>',
            '<ul>': r'</?ul>',
            '<li>': r'</?li>',
        }
        for feat, html_tag_re in html_feature_regexs.items():
            tags = re.finditer(html_tag_re, text, re.IGNORECASE)
            c = sum(1 for _ in tags)
            crel = round(c / len(text) * 1000)
            if c > 0:
                features[feat] = c
            if crel > 0:
                features['relative ' + feat] = crel

        return features

    def score(self, features):
        score = self.model_parameters['bias'] + self.model_parameters.get('threshold', 0)
        weights = self.model_parameters['weights']
        for feat_name, feat_value in features.items():
            score += feat_value * weights.get(feat_name, 0)

        return score

    def predict_raw(self, text):
        return self.score(self.extract_features(text)) - self.model_parameters['bias']

    def predict(self, text):
        return self.score(self.extract_features(text)) < 0
