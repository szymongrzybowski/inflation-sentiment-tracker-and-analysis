from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict, Counter
import random
from dsfs.scratch.machine_learning import split_data
import csv
import pytest
 
def tokenize(text: str) -> Set[str]:
    text = text.lower()
    rem_links = re.sub(r'http\S+', '', text) # deletes links
    all_words = re.findall("[a-z0-9]+", rem_links)
    return set(all_words)
   
class Post(NamedTuple):
    text: str
    is_what: int
 
class SentimentClassifier:
    def __init__(self, k: float = 0.333) -> None:
        self.k = k
 
        self.tokens: Set[str] = set()
        self.token_neg_counts: Dict[str, int] = defaultdict(int)
        self.token_pos_counts: Dict[str, int] = defaultdict(int)
        self.token_neu_counts: Dict[str, int] = defaultdict(int)
        self.neg_posts = self.pos_posts = self.neu_posts = 0
 
    def train(self, posts: Iterable[Post]) -> None:
        for post in posts:
            if post.is_what == 0:
                self.neg_posts += 1
            elif post.is_what == 4:
                self.pos_posts += 1
            elif post.is_what == 2:
                self.neu_posts += 1
            else: 
                break
 
            for token in tokenize(post.text):
                self.tokens.add(token)
                if post.is_what == 0:
                    self.token_neg_counts[token] += 1
                elif post.is_what == 4:
                    self.token_pos_counts[token] += 1
                elif post.is_what == 2:
                    self.token_neu_counts[token] += 1
                else:
                    break

    def _probabilities(self, token: str) -> Tuple[float, float, float]:
        """Returns P(token | neg), P(token | pos) and P(token | neu)"""
        neg = self.token_neg_counts[token]
        pos = self.token_pos_counts[token]
        neu = self.token_neu_counts[token]
 
        p_token_neg = (neg + self.k) / (self.neg_posts + 3 * self.k)
        p_token_pos = (pos + self.k) / (self.pos_posts + 3 * self.k)
        p_token_neu = (neu + self.k) / (self.neu_posts + 3 * self.k)
 
        return p_token_neg, p_token_pos, p_token_neu
 
    def predict(self, text: str) -> List[float]:
        text_tokens = tokenize(text)
        log_prob_if_neg = log_prob_if_pos = log_prob_if_neu = 0.0
 
        for token in self.tokens:
            prob_if_neg, prob_if_pos, prob_if_neu = self._probabilities(token)
 
            if token in text_tokens:
                log_prob_if_neg += math.log(prob_if_neg)
                log_prob_if_pos += math.log(prob_if_pos)
                log_prob_if_neu += math.log(prob_if_neu)
            else:
                log_prob_if_neg += math.log(1.0 - prob_if_neg)
                log_prob_if_pos += math.log(1.0 - prob_if_pos)
                log_prob_if_neu += math.log(1.0 - prob_if_neu)
 
        prob_if_neg = math.exp(log_prob_if_neg)
        prob_if_pos = math.exp(log_prob_if_pos)
        prob_if_neu = math.exp(log_prob_if_neu)
 
        return [prob_if_neg / (prob_if_neg + prob_if_pos + prob_if_neu),
                prob_if_pos / (prob_if_neg + prob_if_pos + prob_if_neu),
                prob_if_neu / (prob_if_neg + prob_if_pos + prob_if_neu)]
 
posts = [Post("neg rules", is_what=0),
         Post("pos rules", is_what=4),
         Post("hello pos", is_what=4),
         Post("neu is the best", is_what=2),
         Post("hello neu", is_what=2)]
 
model = SentimentClassifier(k=0.333)
model.train(posts)

assert model.tokens == {"neg", "pos", "neu", "rules", "hello", "is", "the", "best"}
assert model.neg_posts == 1
assert model.pos_posts == 2
assert model.neu_posts == 2
assert model.token_neg_counts == {"neg": 1, "rules": 1}
assert model.token_pos_counts == {"pos": 2, "rules": 1, "hello": 1}
assert model.token_neu_counts == {"neu": 2, "hello": 1, "is": 1, "the": 1, "best": 1}
 
text = "hello neg"
 
probs_if_neg = [
    (1 + 0.333) / (1 + 3 * 0.333),      # "neg" (present)
    1 - (0 + 0.333) / (1 + 3 * 0.333),  # "pos" (not present)
    1 - (0 + 0.333) / (1 + 3 * 0.333),  # "neu" (not present)
    1 - (1 + 0.333) / (1 + 3 * 0.333),  # "rules" (not present)
    (0 + 0.333) / (1 + 3 * 0.333),      # "hello" (present)
    1 - (0 + 0.333) / (1 + 3 * 0.333),  # "is" (not present)
    1 - (0 + 0.333) / (1 + 3 * 0.333),  # "the" (not present)
    1 - (0 + 0.333) / (1 + 3 * 0.333)   # "best" (not present)
]
 
probs_if_pos = [
    (0 + 0.333) / (2 + 3 * 0.333),      # "neg" (present)
    1 - (2 + 0.333) / (2 + 3 * 0.333),  # "pos" (not present)
    1 - (0 + 0.333) / (2 + 3 * 0.333),  # "neu" (present)
    1 - (1 + 0.333) / (2 + 3 * 0.333),  # "rules" (not present)
    (1 + 0.333) / (2 + 3 * 0.333),      # "hello" (present)
    1 - (0 + 0.333) / (2 + 3 * 0.333),  # "is" (not present)
    1 - (0 + 0.333) / (2 + 3 * 0.333),  # "the" (not present)
    1 - (0 + 0.333) / (2 + 3 * 0.333)   # "best" (not present)
]
 
probs_if_neu = [
    (0 + 0.333) / (2 + 3 * 0.333),      # "neg" (present)
    1 - (0 + 0.333) / (2 + 3 * 0.333),  # "pos" (not present)
    1 - (2 + 0.333) / (2 + 3 * 0.333),  # "neu" (not present)
    1 - (0 + 0.333) / (2 + 3 * 0.333),  # "rules" (not present)
    (1 + 0.333) / (2 + 3 * 0.333),      # "hello" (present)
    1 - (1 + 0.333) / (2 + 3 * 0.333),  # "is" (not present)
    1 - (1 + 0.333) / (2 + 3 * 0.333),  # "the" (not present)
    1 - (1 + 0.333) / (2 + 3 * 0.333)   # "best" (not present)
]
 
p_if_neg = math.exp(sum(math.log(p) for p in probs_if_neg))
p_if_pos = math.exp(sum(math.log(p) for p in probs_if_pos))
p_if_neu = math.exp(sum(math.log(p) for p in probs_if_neu))
 
pytest.approx(model.predict(text)) == [p_if_neg / (p_if_neg + p_if_pos + p_if_neu),
                                       p_if_pos / (p_if_neg + p_if_pos + p_if_neu),
                                       p_if_neu / (p_if_neg + p_if_pos + p_if_neu)]

def main():
    data: List[Post] = []
    filename = './testdata3.csv'
 
    with open(filename, 'r', newline='', encoding='cp437') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for line in csv_reader:
            polarity = int(line[0])
            tweet_text = line[1]
            data.append(Post(tweet_text, polarity))
 
    train_posts, test_posts = split_data(data, 0.75)
   
    model = SentimentClassifier()
    model.train(train_posts)
   
    predictions = [(post, model.predict(post.text))
                   for post in test_posts]

    counter : Dict[str, int] = defaultdict(int)

    for post, prob in predictions:
        if post.is_what == 0:
            if prob[0] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1
        elif post.is_what == 4:
            if prob[1] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1   
        elif post.is_what == 2:
            if prob[2] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1
        else:
            break

    number_of_correct = counter["correct"]
    number_of_wrong = counter["wrong"]

    result = (number_of_correct / (number_of_correct + number_of_wrong)) * 100
    print(result)

if __name__ == "__main__": main()
