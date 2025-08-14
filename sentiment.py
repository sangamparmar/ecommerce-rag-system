"""
Sentiment analysis module for product reviews.
Handles sentiment classification and filtering of product reviews using HuggingFace transformers.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Tuple
import re
import numpy as np


class SentimentAnalyzer:
    """Handles sentiment analysis for product reviews."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the HuggingFace model for sentiment analysis
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.labels_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        
        # Initialize the sentiment analysis pipeline
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model."""
        try:
            print(f"Loading sentiment analysis model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            print("Falling back to default model...")
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis", return_all_scores=True)
                print("Default sentiment model loaded successfully")
            except Exception as e2:
                print(f"Error loading default model: {e2}")
                self.sentiment_pipeline = None
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean and preprocess review text.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned review text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def is_valid_review(self, text: str) -> bool:
        """
        Check if a review is valid (not spam or too short).
        
        Args:
            text: Review text to validate
            
        Returns:
            True if review is valid, False otherwise
        """
        if not isinstance(text, str):
            return False
        
        # Clean the text first
        cleaned_text = self.clean_review_text(text)
        
        # Check minimum length (at least 3 words)
        words = cleaned_text.split()
        if len(words) < 3:
            return False
        
        # Check for repetitive patterns (spam detection)
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
            return False
        
        # Check for excessive caps (likely spam)
        caps_ratio = sum(1 for c in cleaned_text if c.isupper()) / max(len(cleaned_text), 1)
        if caps_ratio > 0.7:  # More than 70% uppercase
            return False
        
        # Check for common spam patterns
        spam_patterns = [
            r'buy\s+now',
            r'click\s+here',
            r'free\s+shipping',
            r'limited\s+time',
            r'\$\$\$+',
            r'www\.',
            r'http'
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                return False
        
        return True
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.sentiment_pipeline:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'error': 'Sentiment model not available'
            }
        
        # Clean and validate text
        cleaned_text = self.clean_review_text(text)
        if not self.is_valid_review(cleaned_text):
            return {
                'sentiment': 'invalid',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
                'error': 'Invalid review text'
            }
        
        try:
            # Get sentiment scores
            results = self.sentiment_pipeline(cleaned_text[:512])  # Limit text length
            
            # Process results based on model output format
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Model returns all scores
                    scores_list = results[0]
                    scores = {}
                    max_score = 0
                    predicted_sentiment = 'neutral'
                    
                    for score_dict in scores_list:
                        label = score_dict['label']
                        score = score_dict['score']
                        
                        # Map label to standard format
                        if label in self.labels_mapping:
                            sentiment_label = self.labels_mapping[label]
                        elif label.lower() in ['positive', 'negative', 'neutral']:
                            sentiment_label = label.lower()
                        else:
                            # Try to infer from label
                            if 'pos' in label.lower():
                                sentiment_label = 'positive'
                            elif 'neg' in label.lower():
                                sentiment_label = 'negative'
                            else:
                                sentiment_label = 'neutral'
                        
                        scores[sentiment_label] = score
                        
                        if score > max_score:
                            max_score = score
                            predicted_sentiment = sentiment_label
                    
                    return {
                        'sentiment': predicted_sentiment,
                        'confidence': max_score,
                        'scores': scores,
                        'cleaned_text': cleaned_text
                    }
                else:
                    # Model returns single prediction
                    result = results[0]
                    sentiment = result['label'].lower()
                    confidence = result['score']
                    
                    # Map to standard labels if needed
                    if sentiment in self.labels_mapping.values():
                        predicted_sentiment = sentiment
                    elif 'pos' in sentiment:
                        predicted_sentiment = 'positive'
                    elif 'neg' in sentiment:
                        predicted_sentiment = 'negative'
                    else:
                        predicted_sentiment = 'neutral'
                    
                    # Create scores dict
                    scores = {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
                    scores[predicted_sentiment] = confidence
                    
                    return {
                        'sentiment': predicted_sentiment,
                        'confidence': confidence,
                        'scores': scores,
                        'cleaned_text': cleaned_text
                    }
            else:
                raise ValueError("Unexpected model output format")
                
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'error': str(e),
                'cleaned_text': cleaned_text
            }
    
    def analyze_reviews_batch(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of reviews.
        
        Args:
            reviews: List of review texts
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for review in reviews:
            if isinstance(review, str):
                result = self.analyze_sentiment(review)
                result['original_text'] = review
                results.append(result)
            else:
                results.append({
                    'sentiment': 'invalid',
                    'confidence': 0.0,
                    'scores': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
                    'error': 'Non-string input',
                    'original_text': str(review)
                })
        
        return results
    
    def get_sentiment_summary(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Get overall sentiment summary for a list of reviews.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        if not reviews:
            return {
                'total_reviews': 0,
                'valid_reviews': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'average_confidence': 0.0,
                'overall_sentiment': 'neutral'
            }
        
        # Analyze all reviews
        results = self.analyze_reviews_batch(reviews)
        
        # Calculate statistics
        valid_results = [r for r in results if r['sentiment'] != 'invalid']
        
        if not valid_results:
            return {
                'total_reviews': len(reviews),
                'valid_reviews': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'average_confidence': 0.0,
                'overall_sentiment': 'neutral'
            }
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_confidence = 0
        
        for result in valid_results:
            sentiment = result['sentiment']
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            total_confidence += result['confidence']
        
        # Calculate percentages
        total_valid = len(valid_results)
        sentiment_percentages = {
            sentiment: (count / total_valid) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Determine overall sentiment
        if sentiment_percentages['positive'] > sentiment_percentages['negative']:
            if sentiment_percentages['positive'] > 50:
                overall_sentiment = 'positive'
            else:
                overall_sentiment = 'mixed_positive'
        elif sentiment_percentages['negative'] > sentiment_percentages['positive']:
            if sentiment_percentages['negative'] > 50:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'mixed_negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'total_reviews': len(reviews),
            'valid_reviews': total_valid,
            'sentiment_distribution': sentiment_percentages,
            'sentiment_counts': sentiment_counts,
            'average_confidence': total_confidence / total_valid if total_valid > 0 else 0,
            'overall_sentiment': overall_sentiment,
            'detailed_results': results
        }
    
    def filter_reviews_by_sentiment(self, reviews: List[str], 
                                   target_sentiment: str = 'positive',
                                   min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter reviews by sentiment and confidence threshold.
        
        Args:
            reviews: List of review texts
            target_sentiment: Target sentiment to filter ('positive', 'negative', 'neutral')
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of filtered review results
        """
        results = self.analyze_reviews_batch(reviews)
        
        filtered_results = []
        for result in results:
            if (result['sentiment'] == target_sentiment and 
                result['confidence'] >= min_confidence):
                filtered_results.append(result)
        
        return filtered_results


def main():
    """Test the SentimentAnalyzer functionality."""
    print("Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    # Test reviews with different sentiments
    test_reviews = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible quality, broke after one week. Waste of money.",
        "It's okay, nothing special but does the job.",
        "wow so good!!!! buy now",  # Should be filtered as spam
        "good",  # Should be filtered as too short
        "Great laptop! Super fast and great for gaming. The build quality is excellent.",
        "Battery life could be better, but performance is outstanding. Highly recommended!",
        "Not suitable for gaming but excellent for office work and studying."
    ]
    
    print("\n=== Testing Individual Review Analysis ===")
    for i, review in enumerate(test_reviews):
        result = analyzer.analyze_sentiment(review)
        print(f"\nReview {i+1}: \"{review}\"")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\n=== Testing Batch Analysis ===")
    batch_results = analyzer.analyze_reviews_batch(test_reviews)
    valid_count = sum(1 for r in batch_results if r['sentiment'] != 'invalid')
    print(f"Processed {len(test_reviews)} reviews, {valid_count} valid")
    
    print("\n=== Testing Sentiment Summary ===")
    summary = analyzer.get_sentiment_summary(test_reviews)
    print(f"Total reviews: {summary['total_reviews']}")
    print(f"Valid reviews: {summary['valid_reviews']}")
    print(f"Overall sentiment: {summary['overall_sentiment']}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    
    print("\n=== Testing Sentiment Filtering ===")
    positive_reviews = analyzer.filter_reviews_by_sentiment(test_reviews, 'positive', 0.5)
    print(f"Found {len(positive_reviews)} positive reviews with confidence >= 0.5")
    for review in positive_reviews:
        print(f"  - \"{review['original_text'][:50]}...\" (confidence: {review['confidence']:.3f})")


if __name__ == "__main__":
    main()
