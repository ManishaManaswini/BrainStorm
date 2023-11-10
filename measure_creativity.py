from time import sleep

import nltk
import pronouncing
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk import bigrams, pos_tag, word_tokenize, sent_tokenize
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import pandas as pd
import os
import numpy as np
import textstat
from load_data import LoadData
#
# # Download necessary NLTK resources
#
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')

# Load the Brown Corpus
brown_words = brown.words()

linguistic_features_weights = {
    "sentence_fluency": 0.15,
    "sentence_novelty": 0.15,
    "sentence_elaboration": 0.1,
    "word_infrequency": 0.1,
    "word_combination_infrequency": 0.1,
    "word_uniqueness": 0.2,
    "syntax_uniqueness_score": 0.1
}

class MeasCre:
    def __init__(self):
        # Calculate bigram frequencies
        self.bigram_frequencies = Counter(bigrams(brown_words))

        # Calculate word frequencies
        self.word_frequencies = Counter(brown_words)

    def is_english(self, s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalpha() and self.is_english(word)]
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]
        return words

    # Function to calculate general word infrequency score
    def general_word_infrequency(self, words):

        # For each word, find its frequency in the corpus
        frequencies = [self.word_frequencies[word] for word in words]

        # Calculate the score based on the frequencies
        # The score is the average frequency
        score = sum(frequencies) / len(frequencies) if frequencies else 0
        return score

    # Function to calculate word combination infrequency score
    def word_combination_infrequency(self, words):

        # Generate bigrams from the words
        bigrams = list(nltk.bigrams(words))

        # For each word, find its frequency in the corpus
        frequencies = [self.bigram_frequencies[bigram] for bigram in bigrams]

        # Calculate the score based on the frequencies
        # The score is the average frequency
        score = sum(frequencies) / len(frequencies) if frequencies else 0
        return score


    # Function to calculate context-specific word uniqueness score
    def context_specific_word_uniqueness(self, words):

        # Count the number of unique words
        unique_words = len(set(words))

        # Calculate the score based on the number of unique words
        # The score is the ratio of unique words to total words
        score = unique_words / len(words) if words else 0

        return score


    # Function to calculate syntax uniqueness score
    def syntax_uniqueness(self, words):

        # Tag each word with its part of speech
        tagged_words = pos_tag(words)

        # Count the number of unique part-of-speech tags
        unique_tags = len(set(tag for word, tag in tagged_words))

        # Calculate the score based on the number of unique tags
        # The score is the ratio of unique tags to total words
        score = unique_tags / len(words) if words else 0

        return score


    # Function to calculate rhyme score between two sentences
    def rhyme_sequence(self, sentence1, sentence2):
        # Tokenize the sentences into words
        words1 = self.preprocess_text(sentence1)
        words2 = self.preprocess_text(sentence2)

        # For each word, find its phonemes using the pronouncing library
        phonemes1 = [pronouncing.phones_for_word(word)[0] for word in words1 if pronouncing.phones_for_word(word)]
        phonemes2 = [pronouncing.phones_for_word(word)[0] for word in words2 if pronouncing.phones_for_word(word)]

        # Check if the last phonemes of the sentences rhyme with each other
        # Here, we'll define rhyming as the last phoneme being the same
        score = phonemes1[-1][-1] == phonemes2[-1][-1] if phonemes1 and phonemes2 else 0

        return score


    # Function to calculate phonetic similarity score between two sentences
    def phonetic_similarity(self, sentence1, sentence2):
        # Tokenize the sentences into words
        words1 = self.preprocess_text(sentence1)
        words2 = self.preprocess_text(sentence2)

        # For each word, find its phonemes using the pronouncing library
        phonemes1 = [pronouncing.phones_for_word(word)[0] for word in words1 if pronouncing.phones_for_word(word)]
        phonemes2 = [pronouncing.phones_for_word(word)[0] for word in words2 if pronouncing.phones_for_word(word)]

        # Compare the phonemes of the sentences to each other
        # Here, we'll define the similarity as the ratio of matching phonemes to total phonemes
        similarities = [
            sum(1 for ph1, ph2 in zip(phonemes1[i], phonemes2[i]) if ph1 == ph2) / max(len(phonemes1[i]), len(phonemes2[i]))
            for i in range(min(len(phonemes1), len(phonemes2)))]

        # Calculate the score based on the phonetic similarity of the sentences
        # The score is the average of the similarities
        score = sum(similarities) / len(similarities) if similarities else 0

        return score


    # Function to calculate sequence similarity score
    def sequence_similarity(self, sentence1, sentence2):
        # Calculate the score based on the sequence similarity of the sentences
        score = SequenceMatcher(None, sentence1, sentence2).ratio()

        return score


    # Function to calculate semantic similarity score
    def semantic_similarity(self, sentence1, sentence2):
        # Tokenize the sentences into words
        words1 = self.preprocess_text(sentence1)
        words2 = self.preprocess_text(sentence2)

        # For each pair of words, find their semantic similarity
        similarities = []
        for word1 in words1:
            for word2 in words2:
                synsets1 = wn.synsets(word1)
                synsets2 = wn.synsets(word2)
                if synsets1 and synsets2:
                    similarity = synsets1[0].path_similarity(synsets2[0])
                    if similarity is not None:
                        similarities.append(similarity)
        # Calculate the score based on the semantic similarities
        # The score is the average of the similarities
        score = sum(similarities) / len(similarities) if similarities else 0

        return score

    def calculate_word_weights(self, words):
        """
        Calculate weights for each word based on word-specific features.
        Returns normalized weights that sum up to 1.
        """
        # Calculate word-specific features
        infrequencies = [self.general_word_infrequency([word]) for word in words]
        uniquenesses = [self.context_specific_word_uniqueness([word]) for word in words]

        # Combine features to calculate raw weights
        raw_weights = [infreq + unique for infreq, unique in zip(infrequencies, uniquenesses)]

        # Normalize weights
        total_weight = sum(raw_weights)
        normalized_weights = [rw / total_weight for rw in raw_weights] if total_weight else [1 / len(words)] * len(
            words)

        return normalized_weights

    # Function to calculate overall creativity score
    def calculate_word_creativity_score(self, text, total_time, epoch_duration=5):
        time = 0
        times = []
        sentence_creativity_scores = []
        sentence_complexity_scores = []
        sentence_readability_scores = []
        text_creativity_score = 0
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        avg_sentence_time = total_time / total_sentences
        for i, sentence in enumerate(sentences):
            words = self.preprocess_text(sentence)
            if len(words) != 0:

                total_words = len(words)
                avg_word_time = avg_sentence_time / total_words

                sentence_fluency = len(words) / (len(words) * avg_word_time)
                sentence_novelty = len(set(words)) / len(words) if words else 0
                avg_sentence_length = len(sentence)
                sentence_elaboration = avg_sentence_length / len(words) if words else 0
                word_infrequency = self.general_word_infrequency(words)
                word_comb_infrequency = self.word_combination_infrequency(words)
                word_uniqueness = self.context_specific_word_uniqueness(words)
                syntax_uniqueness_score = self.syntax_uniqueness(words)
                # Calculate the creativity score for the sentence
                sentence_creativity_score = (
                        linguistic_features_weights['sentence_fluency'] * sentence_fluency +
                        linguistic_features_weights['sentence_novelty'] * sentence_novelty +
                        linguistic_features_weights['sentence_elaboration'] * sentence_elaboration +
                        linguistic_features_weights['word_infrequency'] * word_infrequency +
                        linguistic_features_weights['word_combination_infrequency'] * word_comb_infrequency +
                        linguistic_features_weights['word_uniqueness'] * word_uniqueness +
                        linguistic_features_weights['syntax_uniqueness_score'] * syntax_uniqueness_score
                )

                total_syllables = sum(textstat.syllable_count(word) for word in words)
                flesch_kincaid_complexity = 0.39 * (len(words) / len(sentences)) + 11.8 * (
                        total_syllables / len(words)) - 15.59
                flesch_kincaid_readability = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (
                        total_syllables / len(words))

                if i < len(sentences)-1:

                    text_words = self.preprocess_text(sentences[i]) + self.preprocess_text(sentences[i + 1])
                    text_total_words = len(text_words)
                    text_avg_word_time = (avg_sentence_time*2) / text_total_words
                    # Text fluency: words per minute
                    text_fluency = len(text_words) / (len(text_words) * text_avg_word_time) if words else 0
                    # Text novelty: unique words / total words
                    text_novelty = len(set(text_words)) / len(text_words) if text_words else 0
                    # Text elaboration: text length
                    avg_text_length = len(text_words)
                    text_elaboration = avg_text_length / len(text_words) if text_words else 0
                    text_sequence = (self.sequence_similarity(sentences[i], sentences[i + 1]))
                    text_semantics = (self.semantic_similarity(sentences[i], sentences[i + 1]))
                    text_rhyme = (self.rhyme_sequence(sentences[i], sentences[i + 1]))
                    text_phonetics = (self.phonetic_similarity(sentences[i], sentences[i + 1]))
                    text_creativity_score = (text_fluency+text_novelty+text_elaboration+
                                                 text_sequence+text_semantics+text_rhyme+text_phonetics)

                word_weights = self.calculate_word_weights(words)
                adjusted_creativity_scores = [((sentence_creativity_score * 0.7) + (text_creativity_score * 0.3))
                                              * weight for weight in word_weights]
                adjusted_complexity_scores = [flesch_kincaid_complexity * weight
                                              for weight in word_weights]
                adjusted_readability_scores = [flesch_kincaid_readability * weight
                                              for weight in word_weights]

                for j in range(len(words)):
                    time += avg_word_time
                    sentence_creativity_scores.append(adjusted_creativity_scores[j])
                    sentence_complexity_scores.append(adjusted_complexity_scores[j])
                    sentence_readability_scores.append(adjusted_readability_scores[j])
                    times.append(time)


        creativity_scores = {'data': sentence_creativity_scores, 'times': times}
        # complexity_scores = [x for x in complexity_scores]
        complexity_scores = {'data': sentence_complexity_scores, 'times': times}
        # readability_scores = [x for x in readability_scores]
        readability_scores = {'data': sentence_readability_scores, 'times': times}


        creativity_features = {
            "creativity_scores": creativity_scores,
            "readability_scores": complexity_scores,
            "complexity_scores": readability_scores
        }


        return creativity_features



    def calculate_sent_creativity_score(self, text, total_time, epoch_duration=5):
        time = 0
        times = []
        sentence_creativity_scores = []
        sentence_complexity_scores = []
        sentence_readability_scores = []
        text_creativity_score = 0
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        avg_sentence_time = total_time / total_sentences
        for i, sentence in enumerate(sentences):
            words = self.preprocess_text(sentence)
            if len(words) != 0:

                total_words = len(words)
                avg_word_time = avg_sentence_time / total_words

                sentence_fluency = len(words) / (len(words) * avg_word_time)
                sentence_novelty = len(set(words)) / len(words) if words else 0
                avg_sentence_length = len(sentence)
                sentence_elaboration = avg_sentence_length / len(words) if words else 0
                word_infrequency = self.general_word_infrequency(words)
                word_comb_infrequency = self.word_combination_infrequency(words)
                word_uniqueness = self.context_specific_word_uniqueness(words)
                syntax_uniqueness_score = self.syntax_uniqueness(words)
                # Calculate the creativity score for the sentence
                sentence_creativity_score = (
                        linguistic_features_weights['sentence_fluency'] * sentence_fluency +
                        linguistic_features_weights['sentence_novelty'] * sentence_novelty +
                        linguistic_features_weights['sentence_elaboration'] * sentence_elaboration +
                        linguistic_features_weights['word_infrequency'] * word_infrequency +
                        linguistic_features_weights['word_combination_infrequency'] * word_comb_infrequency +
                        linguistic_features_weights['word_uniqueness'] * word_uniqueness +
                        linguistic_features_weights['syntax_uniqueness_score'] * syntax_uniqueness_score
                )

                total_syllables = sum(textstat.syllable_count(word) for word in words)
                flesch_kincaid_complexity = 0.39 * (len(words) / len(sentences)) + 11.8 * (
                        total_syllables / len(words)) - 15.59
                flesch_kincaid_readability = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (
                        total_syllables / len(words))

                if i < len(sentences)-1:

                    text_words = self.preprocess_text(sentences[i]) + self.preprocess_text(sentences[i + 1])
                    text_total_words = len(text_words)
                    text_avg_word_time = (avg_sentence_time*2) / text_total_words
                    # Text fluency: words per minute
                    text_fluency = len(text_words) / (len(text_words) * text_avg_word_time) if words else 0
                    # Text novelty: unique words / total words
                    text_novelty = len(set(text_words)) / len(text_words) if text_words else 0
                    # Text elaboration: text length
                    avg_text_length = len(text_words)
                    text_elaboration = avg_text_length / len(text_words) if text_words else 0
                    text_sequence = (self.sequence_similarity(sentences[i], sentences[i + 1]))
                    text_semantics = (self.semantic_similarity(sentences[i], sentences[i + 1]))
                    text_rhyme = (self.rhyme_sequence(sentences[i], sentences[i + 1]))
                    text_phonetics = (self.phonetic_similarity(sentences[i], sentences[i + 1]))
                    text_creativity_score = (text_fluency+text_novelty+text_elaboration+
                                                 text_sequence+text_semantics+text_rhyme+text_phonetics)


                time += avg_sentence_time
                sentence_creativity_scores.append((sentence_creativity_score * 0.7) + (text_creativity_score * 0.3))
                sentence_complexity_scores.append(flesch_kincaid_complexity)
                sentence_readability_scores.append(flesch_kincaid_readability)
                times.append(time)


        creativity_scores = {'data': sentence_creativity_scores, 'times': times}
        # complexity_scores = [x for x in complexity_scores]
        complexity_scores = {'data': sentence_complexity_scores, 'times': times}
        # readability_scores = [x for x in readability_scores]
        readability_scores = {'data': sentence_readability_scores, 'times': times}


        creativity_features = {
            "creativity_scores": creativity_scores,
            "readability_scores": complexity_scores,
            "complexity_scores": readability_scores
        }


        return creativity_features


    def normalize_all_creativity_features(self, all_creativity_features):
        scaler = MinMaxScaler()

        for file_name, features in all_creativity_features.items():
            for feature_name, feature_dict in features.items():
                data = np.array(feature_dict['data']).reshape(-1, 1)
                normalized_data = scaler.fit_transform(data).reshape(-1).tolist()
                all_creativity_features[file_name][feature_name]['data'] = normalized_data

        return all_creativity_features


    def plot_all_creativity_features(self, all_creativity_features, parameters):
        keys = list(all_creativity_features.keys())
        figs = []
        titles = []
        for i in range(0, len(keys), 3):  # Loop through files, 3 at a time
            fig, axs = plt.subplots(min(3, len(keys)-i), 1, figsize=(10, 15))  # Create a new figure for each group of 3 files
            for j in range(3):
                if i+j < len(keys):
                    key = keys[i+j]
                    features = all_creativity_features[key]
                    for parameter in parameters:
                        smoothed_scores = self.calculate_moving_average(features[parameter]['data'], 10)
                        axs[j].plot(features[parameter]['times'], smoothed_scores , label=parameter)
                    axs[j].legend()
                    axs[j].set_title(key)
            figs.append(fig)
            titles.append(f"files{i}-{j+i}")
        return figs, titles


    def calculate_moving_average(self, data, window_size):
        smas = pd.Series(data).rolling(window_size, min_periods=1, center=True).mean().to_numpy()
        return smas


    def analyze_creativity_multiple_files(self, folder_path, total_times, type="sent"):
        # Get a list of all text files in the folder
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        all_creativity_features = {}

        for i, file_name in enumerate(file_names):
            # Load the text file
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                text = file.read()

            s = file_name
            c = '_'
            n = [pos for pos, char in enumerate(s) if char == c][1]
            file = file_name[0:n]

            # Calculate creativity scores and times
            if type == "sent":
                all_creativity_features[file] = self.calculate_sent_creativity_score(text, total_times[i])
            else:
                all_creativity_features[file] = self.calculate_word_creativity_score(text, total_times[i])

        parameters = ["creativity_scores", "readability_scores", "complexity_scores"]
        # all_creativity_features = self.normalize_all_creativity_features(all_creativity_features)
        # self.plot_all_creativity_features(all_creativity_features, parameters)
        return all_creativity_features







