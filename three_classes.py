from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict, Counter
import random
from scratch.machine_learning import split_data # https://github.com/joelgrus/data-science-from-scratch/tree/master/scratch

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)
    
class Post(NamedTuple):
    text: str
    is_what: int

class SentimentClassifier:
    def __init__(self, k: float = 0.5) -> None:
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
            else:
                self.neu_posts += 1

            for token in tokenize(post.text):
                self.tokens.add(token)
                if post.is_what == 0:
                    self.token_neg_counts[token] += 1
                elif post.is_what == 4:
                    self.token_pos_counts[token] += 1
                else:
                    self.token_neu_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Returns P(token | neg), P(token | pos) and P(token | neu)"""
        neg = self.token_neg_counts[token]
        pos = self.token_pos_counts[token]
        neu = self.token_neu_counts[token]

        p_token_neg = (neg + self.k) / (self.neg_posts + 2 * self.k)
        p_token_pos = (pos + self.k) / (self.pos_posts + 2 * self.k)
        p_token_neu = (neu + self.k) / (self.neu_posts + 2 * self.k)

        return p_token_neg, p_token_pos, p_token_neu

    def predict(self, text: str) -> float:
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
                prop_if_pos / (prob_if_neg + prob_if_pos + prob_if_neu),
                prob_if_neu / (prob_if_neg + prob_if_pos + prob_if_neu)]

posts = [Post("neg rules", is_what=0),
         Post("pos rules", is_what=4),
         Post("hello pos", is_what=4),
         Post("neu is the best", is_what=2),
         Post("hello neu", is_what=2)]

model = SentimentClassifier(k=0.5)
model.train(posts)

assert model.tokens == {"neg", "pos", "neu", "rules", "hello", "is", "the", "best"}
assert model.neg_posts == 1
assert model.pos_posts == 2
assert model.neu_posts == 2
assert model.token_neg_counts == {"neg": 1, "rules": 1}
assert model.token_pos_counts == {"pos": 2, "rules": 1, "hello": 1}
assert model.token_neu_counts == {"neu": 2, "hello": 1, "is": 1, "the": 1, "best": 1}
